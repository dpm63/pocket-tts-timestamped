"""Train, export and check leading-silence vectors for one checkpoint (one workflow job)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import safetensors.torch
import torch

from pocket_tts_timestamped import AudioChunk, TTSModel
from scripts.leading_silence import data, evaluate, transcribe
from scripts.leading_silence.onsets import VOICED_DB, frame_db
from scripts.leading_silence.sequences import FRAME_SAMPLES, encode_voice
from scripts.leading_silence.steer import Steer
from scripts.leading_silence.train import (
    FullRows,
    Rows,
    Setting,
    default_setting,
    make_full_rows,
    make_rows,
    train,
)
from scripts.leading_silence.weights import METADATA_KEY, weights_sha256

WARMUP_FRAMES = 8  # the decoder warms up on this much encoded silence
ROLLOUT_BATCH = 32  # onset rollouts (32 frames); whole-sequence ones use Setting.full_batch
VECTORS_FILE = "vectors.safetensors"
REPORT_FILE = "report.json"
Phase = Callable[[str, Callable[[], Any]], Any]


def silence_latents(model: TTSModel, frames: int) -> torch.Tensor:
    """Normalized latents [frames, ldim] of Mimi-encoded digital silence, or of the model's
    own generated silence where the encoder can't produce FlowLM latents."""
    flow_lm = model.flow_lm
    zeros = torch.zeros(1, 1, frames * FRAME_SAMPLES, device=flow_lm.bos_emb.device)
    raw = model.mimi.encode_to_latent(zeros)[0]
    if raw.shape[-1] != flow_lm.ldim:
        # English 2026-01: its encoder outputs 512-d features and only the decoder side
        # projects to the 32-d latent, so encoded silence never lands in latent space.
        return generated_silence_latents(model, frames)
    return ((raw - flow_lm.emb_mean) / flow_lm.emb_std).float().contiguous()


GENERATED_SILENCE_VOICES = ("alba", "george", "mary", "paul")  # train-split voices
GENERATED_SILENCE_DB = -65.0  # English 2026-01's own silence sits near -72 dB
# Short texts start instantly on English 2026-01.
GENERATED_SILENCE_TEXT = "The streets were quiet that morning, and nobody was awake yet."
GENERATED_SILENCE_SEEDS = 32
GENERATED_SILENCE_MAX_FRAMES = 40


@torch.no_grad()
def generated_silence_latents(model: TTSModel, frames: int) -> torch.Tensor:
    """`frames` latents [frames, ldim] of the model's own silence: frames before the first
    voiced one in plain generations (train voices, fixed text and seeds) that decode below
    GENERATED_SILENCE_DB. Each generation's first 2 frames (cold decoder) and the frame before
    its onset are skipped (bos_shortcut generated_silence_latents). The RNG is left as it
    was."""
    rng = torch.get_rng_state()
    found: list[torch.Tensor] = []
    states = {voice: model.get_state_for_audio_prompt(voice) for voice in GENERATED_SILENCE_VOICES}
    try:
        for seed in range(GENERATED_SILENCE_SEEDS):
            for state in states.values():
                torch.manual_seed(seed)
                session = model.start_session(state)
                session.set_text(session.prepare_text(GENERATED_SILENCE_TEXT))
                before: list[tuple[torch.Tensor, float]] = []
                for _ in range(GENERATED_SILENCE_MAX_FRAMES):
                    frame = session.generate_frame()
                    (audio,) = [
                        e.audio for e in session.decode([frame]) if isinstance(e, AudioChunk)
                    ]
                    level = float(frame_db(audio.numpy())[0])
                    if level > VOICED_DB:
                        break
                    before.append((frame.latent[0, 0], level))
                found += [latent for latent, level in before[2:-1] if level < GENERATED_SILENCE_DB]
                if len(found) >= frames:
                    return torch.stack(found[:frames]).float().contiguous()
    finally:
        torch.set_rng_state(rng)
    raise RuntimeError(f"Only {len(found)} silent frames before speech in plain generations")


def _stored_bytes(prefix_kv: list[dict[str, torch.Tensor]]) -> int:
    return sum(tensor.nbytes for row in prefix_kv for tensor in row.values())


def _train(
    model: TTSModel, folder: Path, setting: Setting, phase: Phase, sizes: dict[str, float]
) -> Steer:
    """Teacher rollouts on the train split, then training. The rollouts (with every row's
    prefix keys/values, several GB) are freed on return. Records the rows kept and their
    stored keys/values in `sizes`."""
    catalog = data.voices(folder, "train")
    pool = data.texts(folder, "train")
    with torch.no_grad():
        latents = phase(
            "encode_voices",
            lambda: {index: encode_voice(model, v.source) for index, v in enumerate(catalog)},
        )
    pairs = data.sample_pairs(catalog, pool, 4 * setting.rows, seed=1)
    rows: Rows = phase(
        "rollouts", lambda: make_rows(model, latents, pairs, setting.rows, ROLLOUT_BATCH, seed=1)
    )
    full: FullRows | None = None
    if setting.full_rows:
        full_pairs = [
            (voice, evaluate.bare(text))
            for voice, text in data.sample_pairs(catalog, pool, setting.full_pairs, seed=3)
        ]
        model.append_terminal_punctuation = False  # these rows end without a period
        try:
            full = phase(
                "full_rollouts",
                lambda: make_full_rows(
                    model, latents, full_pairs, setting.full_batch, setting.full_max_frames, seed=3
                ),
            )
        finally:
            model.append_terminal_punctuation = model.config.append_terminal_punctuation
    sizes["rows"] = len(rows.k)
    sizes["rows_stored_mb"] = round(_stored_bytes(rows.prefix_kv) / 2**20, 1)
    if full is not None:
        sizes["full_rows"] = len(full.k)
        sizes["full_rows_stored_mb"] = round(_stored_bytes(full.prefix_kv) / 2**20, 1)
    return phase("train", lambda: train(model, rows, full, setting))


def _peak_rss_mb() -> float | None:
    """Peak resident memory of this process so far; None on Windows."""
    if sys.platform == "win32":
        return None
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux, bytes on macOS
    return round(peak / 1024 if sys.platform == "linux" else peak / 2**20, 1)


# --timing runs every phase at this size, with the real batch shapes, and scales its times up.
TIMING_ROWS = 64  # one full training batch
TIMING_STEPS = 3
TIMING_EVAL_TEXTS = 2
TIMING_WHISPER_TRIALS = 6


def _extrapolate(
    times: dict[str, float], timed: Setting, setting: Setting, eval_texts: int, whisper: int
) -> dict[str, float]:
    """Full-run seconds per phase from a --timing run of `timed` instead of `setting`."""
    scale = {
        "rollouts": setting.rows / timed.rows,
        "full_rollouts": setting.full_pairs / timed.full_pairs,
        "train": setting.steps / timed.steps,
    }
    for side in ("baseline", "vectors"):
        for part in ("onsets", "tail_without_period", "tail_with_period"):
            scale[f"{side}_{part}"] = eval_texts / TIMING_EVAL_TEXTS
        scale[f"{side}_whisper"] = whisper / TIMING_WHISPER_TRIALS
    full = {name: round(seconds * scale.get(name, 1.0), 1) for name, seconds in times.items()}
    full["total"] = round(sum(full.values()), 1)
    return full


def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads or os.cpu_count() or 1)
    times: dict[str, float] = {}
    peak_rss_mb: dict[str, float | None] = {}

    def phase(name: str, call: Callable[[], Any]) -> Any:  # noqa: ANN401
        started = time.perf_counter()
        result = call()
        times[name] = round(time.perf_counter() - started, 1)
        peak_rss_mb[name] = _peak_rss_mb()
        print(f"[{name}] {times[name]:.1f} s, peak RSS {peak_rss_mb[name]} MB", flush=True)
        return result

    model = phase("load", lambda: TTSModel.load_model(language=args.config))
    if not model.has_voice_cloning:
        raise SystemExit("Training needs the voice-cloning weights (set HF_TOKEN)")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    language = data.language_of(args.config)
    folder = phase("data", lambda: data.fetch(language))
    full_setting = setting = default_setting(len(model.flow_lm.transformer.layers))
    eval_texts, whisper_trials = args.eval_texts, args.whisper_trials
    if args.smoke:
        setting = dataclasses.replace(setting, rows=16, steps=4, full_pairs=16, full_max_frames=120)
    if args.timing:
        setting = dataclasses.replace(
            setting, rows=TIMING_ROWS, steps=TIMING_STEPS, full_pairs=setting.full_batch
        )
        eval_texts, whisper_trials = TIMING_EVAL_TEXTS, TIMING_WHISPER_TRIALS
    sizes: dict[str, float] = {}
    steer = _train(model, folder, setting, phase, sizes)

    path = out / VECTORS_FILE
    vectors = steer.vectors().detach().float().contiguous()
    with torch.no_grad():
        warmup = silence_latents(model, WARMUP_FRAMES)
    safetensors.torch.save_file(
        {"vectors": vectors.cpu(), "warmup_latents": warmup.cpu()},
        str(path),
        metadata={
            "config": args.config,
            "setting": json.dumps(dataclasses.asdict(setting)),
            METADATA_KEY: weights_sha256(model.config.weights_path),
        },
    )

    test_voices = data.voices(folder, "test")
    test_texts = data.texts(folder, "test")[:eval_texts]
    with torch.no_grad():
        states = [model.get_state_for_audio_prompt(voice.source) for voice in test_voices]
    whisper = None
    if whisper_trials:
        whisper = phase("load_whisper", lambda: transcribe.load_whisper(torch.get_num_threads()))

    def check(side: str) -> dict[str, Any]:
        return evaluate.measure(
            model,
            states,
            test_texts,
            whisper=whisper,
            language=language,
            whisper_trials=whisper_trials,
            phase=lambda name, call: phase(f"{side}_{name}", call),
        )

    baseline = check("baseline")
    model._load_leading_silence_vectors(path)
    steered = check("vectors")
    report = {
        "config": args.config,
        "setting": dataclasses.asdict(setting),
        "smoke": args.smoke,
        "timing": args.timing,
        "test_voices": [voice.id for voice in test_voices],
        "test_texts": len(test_texts),
        "baseline": baseline,
        "vectors": steered,
        "problems": evaluate.problems(baseline, steered),
        "times_s": times,
        "peak_rss_mb": peak_rss_mb,
        "sizes": sizes,
        "torch_threads": torch.get_num_threads(),
    }
    if args.timing:
        report["full_run_times_s"] = _extrapolate(
            times, setting, full_setting, args.eval_texts, args.whisper_trials
        )
        # Stored keys/values grow with the rows kept; the rest of the peak doesn't.
        report["full_run_stored_mb"] = round(
            sizes["rows_stored_mb"] * full_setting.rows / setting.rows
            + sizes.get("full_rows_stored_mb", 0) * full_setting.full_pairs / setting.full_pairs,
            1,
        )
    (out / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
