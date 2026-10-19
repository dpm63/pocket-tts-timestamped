"""Synchronous generation with an optional steer, onset measurement, whisper scoring."""

import copy
import re

import numpy as np
import torch
from pocket_tts.modules.stateful_module import increment_steps, init_states

from .sequences import FRAME_SAMPLES, prepare_text, text_tokens
from .sites import Steer, steered

SAMPLE_RATE = 24000
THRESHOLD_DB = -60.0  # per 80 ms frame
MIN_RUN = 3  # voiced frames in a row that count as speech onset
WHISPER_LEAD_PAD_S = 0.3  # whisper misses words that start at sample 0


def frame_db(pcm: np.ndarray) -> np.ndarray:
    n = len(pcm) // FRAME_SAMPLES
    frames = pcm[: n * FRAME_SAMPLES].reshape(n, FRAME_SAMPLES).astype(np.float64)
    return 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-12)


def onset(voiced: np.ndarray, min_run: int = MIN_RUN) -> tuple[int, int]:
    """(first voiced frame, start of the first `min_run` voiced frames); -1 if absent."""
    first = int(np.argmax(voiced)) if voiced.any() else -1
    for i in range(len(voiced) - min_run + 1):
        if voiced[i : i + min_run].all():
            return first, i
    return first, -1


START_FRAMES, BODY_FROM = 2, 4  # onset loudness: first 160 ms vs the body from 320 ms on


SPEECH_THRESHOLD_DB = -35.0  # onset of speech proper, past breaths and room noise


def onset_loudness(audio: np.ndarray, threshold_db: float = SPEECH_THRESHOLD_DB) -> dict | None:
    """Loudness of the first speech frames relative to the rest of the utterance.

    start_rel_db: RMS dB of the first 160 ms after sustained onset (at
    `threshold_db`; the -60 dB silence threshold would start at breaths) minus
    the median 80 ms-frame dB of the voiced body. peak_rel_db: peak |x| in
    those 160 ms vs the body's 99th-percentile |x|.
    """
    db = frame_db(audio)
    _, k = onset(db > threshold_db)
    if k < 0 or len(db) < k + BODY_FROM + 4:
        return None
    body = db[k + BODY_FROM :]
    body = body[body > -40]
    start = audio[k * FRAME_SAMPLES : (k + START_FRAMES) * FRAME_SAMPLES]
    rest = audio[(k + BODY_FROM) * FRAME_SAMPLES :]
    return {
        "onset": k,
        "start_rel_db": float(20 * np.log10(np.sqrt(np.mean(start**2)) + 1e-12) - np.median(body)),
        "peak_rel_db": float(
            20 * np.log10((np.abs(start).max() + 1e-12) / (np.quantile(np.abs(rest), 0.99) + 1e-12))
        ),
    }


HEAD_MS = 30
ABRUPT_DB = -45.0  # onset head louder than this = abrupt start (the step-100 artifact)


def onset_head_db(audio: np.ndarray, onset_frame: int, ms: int = HEAD_MS) -> float:
    """RMS dB of the first `ms` of the first speech frame.

    Natural onsets (the teacher's frame k) begin that frame near-silent and
    ramp up; an abrupt start puts low-level noise right at the frame start.
    """
    start = onset_frame * FRAME_SAMPLES
    head = audio[start : start + ms * SAMPLE_RATE // 1000]
    return float(20 * np.log10(np.sqrt(np.mean(head**2)) + 1e-9)) if len(head) else float("nan")


@torch.no_grad()
def silence_latents(model, n_frames: int) -> torch.Tensor:
    """Normalized flow-LM latents [n, C] of digital silence (Mimi-encoded zeros)."""
    fl = model.flow_lm
    zeros = torch.zeros(1, 1, n_frames * FRAME_SAMPLES, device=fl.bos_emb.device)
    raw = model.mimi.encode_to_latent(zeros)[0]
    return (raw - fl.emb_mean) / fl.emb_std


@torch.no_grad()
def warm_mimi_state(model, warm_latents: torch.Tensor | None, n_frames: int) -> dict:
    """Mimi decoder state that has already decoded `warm_latents` (normalized [W, C]).

    The state only depends on the warm-up latents, so in deployment it is
    computed once and copied per request.
    """
    fl, mimi = model.flow_lm, model.mimi
    steps_per_latent = int(mimi.encoder_frame_rate / mimi.frame_rate)
    n_warm = 0 if warm_latents is None else warm_latents.shape[0]
    state = init_states(mimi, batch_size=1, sequence_length=(n_warm + n_frames) * steps_per_latent)
    if n_warm:
        mimi.decode_from_latent(warm_latents[None] * fl.emb_std + fl.emb_mean, state)
        increment_steps(mimi, state, increment=n_warm * steps_per_latent)
    return state


@torch.no_grad()
def generate_sync(
    model,
    voice_state: dict,
    text: str,
    steer: Steer | None = None,
    stop_at_onset: bool = False,
    max_frames: int | None = None,
    mimi_warmup: torch.Tensor | None = None,
) -> dict:
    """Frame-by-frame generation on this thread (TTSModel's stream runs daemon
    threads that keep going after the consumer stops and share the RNG).

    stop_at_onset: stop once MIN_RUN voiced frames in a row appeared.
    Otherwise run to EOS + frames_after_eos like TTSModel.generate_audio.
    mimi_warmup: normalized latents the Mimi decoder decodes (and discards) first.
    """
    state = copy.deepcopy(voice_state)
    prepared_text, frames_after_eos = prepare_text(model, text)
    frames_after_eos += 2
    tokens = text_tokens(model, prepared_text)
    n_frames = model._estimate_max_gen_len(tokens.shape[1]) if max_frames is None else max_frames
    model._expand_kv_cache(
        state, sequence_length=model._flow_lm_current_end(state) + tokens.shape[1] + 1 + n_frames
    )
    steps_per_latent = int(model.mimi.encoder_frame_rate / model.mimi.frame_rate)
    mimi_state = warm_mimi_state(model, mimi_warmup, n_frames)
    fl = model.flow_lm
    with steered(model, steer):
        soft = (
            steer.vector()[None, None].to(fl.dtype)
            if steer is not None and steer.site.inserts
            else None
        )
        model._run_flow_lm_and_increment_step(
            model_state=state, text_tokens=tokens, audio_conditioning=soft
        )
        latent = torch.full((1, 1, fl.ldim), float("nan"), device=fl.bos_emb.device, dtype=fl.dtype)
        frames, voiced, latents, eos_step = [], [], [], None
        for step in range(n_frames):
            latent, is_eos = model._run_flow_lm_and_increment_step(
                model_state=state, backbone_input_latents=latent
            )
            if is_eos.item() and eos_step is None:
                eos_step = step
            if eos_step is not None and step >= eos_step + frames_after_eos:
                break
            pcm = model.mimi.decode_from_latent(latent * fl.emb_std + fl.emb_mean, mimi_state)
            increment_steps(model.mimi, mimi_state, increment=steps_per_latent)
            frame = pcm.float().cpu().numpy().ravel()
            frames.append(frame)
            latents.append(latent[0, 0])
            voiced.append(frame_db(frame)[0] > THRESHOLD_DB)
            if stop_at_onset and len(voiced) >= MIN_RUN and all(voiced[-MIN_RUN:]):
                break
    first, sustained = onset(np.array(voiced))
    return {
        "audio": np.concatenate(frames) if frames else np.zeros(0, np.float32),
        "first_voiced": first,
        "sustained": sustained,
        "eos_step": -1 if eos_step is None else eos_step,
        "latents": torch.stack(latents)
        if latents
        else torch.zeros(0, fl.ldim),  # normalized [T, C]
    }


def measure_onsets(
    model,
    steer: Steer | None,
    states: dict,
    texts: list[str],
    mimi_warmup: torch.Tensor | None = None,
) -> dict:
    """Frame-0 rate and onset-artifact level over onset trials (one paired seed per text).

    states: {voice id: voice state}, computed with the steer applied when its site is cacheable.
    """
    heads, frame0, n = [], 0, 0
    for state in states.values():
        for text_id, text in enumerate(texts):
            torch.manual_seed(1_000 * text_id)
            r = generate_sync(
                model,
                state,
                text,
                steer,
                stop_at_onset=True,
                max_frames=40,
                mimi_warmup=mimi_warmup,
            )
            n += 1
            if r["sustained"] < 0:
                continue
            frame0 += r["sustained"] == 0
            heads.append(onset_head_db(r["audio"], r["sustained"]))
    heads = np.array(heads)
    return {
        "n_trials": n, "frame0": int(frame0), "frame0_rate": round(frame0 / n, 3),
        "median_head_db": round(float(np.median(heads)), 1), "abrupt_rate": round(float((heads > ABRUPT_DB).mean()), 3),
    }  # fmt: skip


def evaluate_generation(model, steer: Steer | None, voices: list, texts: list[str], reps: int, num_full: int, whisper, out_dir=None, mimi_warmup: torch.Tensor | None = None,
                        language: str = "en") -> tuple[list[dict], list[dict]]:  # fmt: skip
    """Onset trials (paired seeds across steers) and full, transcribed generations (whisper `language`)."""
    import soundfile as sf

    onset_rows, full_rows = [], []
    for voice in voices:
        cache_in_state = steer is not None and steer.site.cacheable
        with steered(model, steer if cache_in_state else None):
            state = model.get_state_for_audio_prompt(voice.source)
        for text_id, text in enumerate(texts):
            for rep in range(reps):
                torch.manual_seed(1_000 * text_id + rep)
                r = generate_sync(
                    model,
                    state,
                    text,
                    steer,
                    stop_at_onset=True,
                    max_frames=40,
                    mimi_warmup=mimi_warmup,
                )
                onset_rows.append(
                    {
                        "voice": voice.id,
                        "split": voice.split,
                        "text_id": text_id,
                        "rep": rep,
                        "first_voiced": r["first_voiced"],
                        "sustained": r["sustained"],
                    }
                )
        for text_id, text in enumerate(texts[:num_full]):
            torch.manual_seed(1_000 * text_id)
            r = generate_sync(model, state, text, steer, mimi_warmup=mimi_warmup)
            loud = onset_loudness(r["audio"]) or {
                "start_rel_db": float("nan"),
                "peak_rel_db": float("nan"),
            }
            if out_dir is not None:
                path = out_dir / voice.id / f"text{text_id}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(path), r["audio"], SAMPLE_RATE, subtype="PCM_16")
            full_rows.append(
                {"voice": voice.id, "split": voice.split, "text_id": text_id, "first_voiced": r["first_voiced"], "sustained": r["sustained"],
                 "duration_s": round(len(r["audio"]) / SAMPLE_RATE, 2), "eos_step": r["eos_step"],
                 "start_rel_db": round(loud["start_rel_db"], 2), "peak_rel_db": round(loud["peak_rel_db"], 2), **transcribe_and_score(whisper, r["audio"], text, language)}
            )  # fmt: skip
        print(f"  {voice.id}: done", flush=True)
    return onset_rows, full_rows


def words_of(text: str) -> list[str]:
    # Letters of any script (same as [a-z']+ on ASCII English).
    return re.findall(r"(?:[^\W\d_]|')+", text.lower().replace("’", "'"))


def edit_distance(a: list[str], b: list[str]) -> int:
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return d[len(b)]


def transcribe_and_score(whisper, audio: np.ndarray, text: str, language: str = "en") -> dict:
    """Whisper on the generation (with a short silence in front) vs the requested text."""
    padded = np.concatenate(
        [np.zeros(int(WHISPER_LEAD_PAD_S * SAMPLE_RATE), np.float32), audio.astype(np.float32)]
    )
    # faster-whisper takes 16 kHz mono float arrays.
    resampled = np.interp(
        np.arange(0, len(padded), SAMPLE_RATE / 16000), np.arange(len(padded)), padded
    ).astype(np.float32)
    segments, _ = whisper.transcribe(resampled, language=language)
    hyp = words_of(" ".join(seg.text for seg in segments))
    ref = words_of(text)
    return {
        # Words before the text's first word = leading junk; -1 = first word missing.
        "lead_extra_words": hyp.index(ref[0]) if ref[0] in hyp[:4] else -1,
        "first_word_ok": bool(hyp[:1] == ref[:1]),
        "wer": round(edit_distance(ref, hyp) / len(ref), 3),
        "transcript": " ".join(hyp),
    }
