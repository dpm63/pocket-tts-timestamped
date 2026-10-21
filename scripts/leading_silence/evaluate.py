"""Check trained vectors through the fork's own generation (`Session`), without and with the
exported file loaded: the onset they buy, and the harm they may do later in the utterance."""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from pocket_tts_timestamped import AudioChunk, TTSModel
from pocket_tts_timestamped.modules.stateful_module import ModelState
from scripts.leading_silence.onsets import (
    ABRUPT_DB,
    VOICED_DB,
    frame_db,
    head_db,
    speech_onset,
    sustained,
)

ONSET_FRAMES = 40  # an onset trial gives up (and counts the onset as 40) after this
ONSET_TAIL = 4  # frames generated past the onset
# A frame after EOS this loud is speech: a stray word or a cut one. Quieter tails above
# TAIL_SOUND_DB are mostly the decay of the last word, which isn't audible as a problem
# (listening, 2026-10-22; pocket-tts-streaming/listening/leading-silence-tails).
TAIL_SPEECH_DB = -35.0
TAIL_SOUND_DB = -50.0
FINAL_PUNCTUATION = ".!?,;:\"')»”…"


def bare(text: str) -> str:
    """The text without its final punctuation, as a mid-text streaming window ends."""
    return text.rstrip().rstrip(FINAL_PUNCTUATION).rstrip()


def _frames(
    model: TTSModel,
    state: ModelState,
    text: str,
    period: bool,
    stop: Callable[[NDArray[np.float32]], bool],
) -> tuple[NDArray[np.float32], int | None]:
    """PCM generated for `text` until `stop(pcm)` or EOS plus its tail; and the EOS frame."""
    session = model.start_session(state)
    window = session.prepare_text(text, append_terminal_punctuation=period)
    session.set_text(window)
    chunks: list[NDArray[np.float32]] = []
    eos = None
    for step in range(400):
        frame = session.generate_frame()
        if frame.eos and eos is None:
            eos = step
        if eos is not None and step >= eos + window.frames_after_eos:
            break
        audio = [e.audio for e in session.decode([frame]) if isinstance(e, AudioChunk)]
        chunks.append(audio[0].numpy())
        if stop(np.concatenate(chunks)):
            break
    return np.concatenate(chunks), eos


def onsets(model: TTSModel, states: Iterable[ModelState], texts: list[str]) -> dict[str, float]:
    """Speech onset (`speech_onset`, -35 dB) per voice x text, one seed per text."""
    frames, heads = [], []
    for state in states:
        for index, text in enumerate(texts):
            torch.manual_seed(1000 * index)

            def stop(pcm: NDArray[np.float32]) -> bool:
                found = speech_onset(frame_db(pcm))
                count = len(pcm) // 1920
                return count >= ONSET_FRAMES or (found >= 0 and count > found + ONSET_TAIL)

            pcm, _ = _frames(model, state, text, True, stop)
            db = frame_db(pcm)
            found = speech_onset(db)
            frames.append(found if found >= 0 else ONSET_FRAMES)
            start = sustained(db > VOICED_DB)
            if start >= 0:
                heads.append(head_db(pcm, start, model.sample_rate))
    array = np.array(frames)
    return {
        "trials": len(frames),
        "speech_by_frame_1": round(float((array <= 1).mean()), 3),
        "onset_median": float(np.median(array)),
        "onset_p95": float(np.percentile(array, 95)),
        "onset_max": int(array.max()),
        "abrupt_rate": round(float((np.array(heads) > ABRUPT_DB).mean()), 3),
        "head_db_median": round(statistics.median(heads), 1),
    }


def tails(
    model: TTSModel, states: Iterable[ModelState], texts: list[str], period: bool
) -> dict[str, float]:
    """Whole generations to EOS plus the tail: how often the tail holds speech (or any sound),
    and how often EOS fires on the last loud frame, which shortens the ending."""
    speech, sound, early, no_eos, count = 0, 0, 0, 0, 0
    for state in states:
        for index, text in enumerate(texts):
            torch.manual_seed(1000 * index + 7)
            pcm, eos = _frames(
                model, state, text if period else bare(text), period, lambda _: False
            )
            count += 1
            if eos is None:
                no_eos += 1
                continue
            db = frame_db(pcm)
            speech += bool((db[eos + 1 :] > TAIL_SPEECH_DB).any())
            sound += bool((db[eos + 1 :] > TAIL_SOUND_DB).any())
            loud = np.flatnonzero(db[: eos + 1] > TAIL_SPEECH_DB)
            early += bool(len(loud)) and int(loud[-1]) >= eos
    ended = max(1, count - no_eos)
    return {
        "trials": count,
        "no_eos": no_eos,
        "tail_speech_rate": round(speech / ended, 3),
        "tail_sound_rate": round(sound / ended, 3),
        "early_eos_rate": round(early / ended, 3),
    }


def measure(
    model: TTSModel, states: list[ModelState], texts: list[str]
) -> dict[str, dict[str, float]]:
    with torch.no_grad():
        return {
            "onsets": onsets(model, states, texts),
            "tail_without_period": tails(model, states, texts, period=False),
            "tail_with_period": tails(model, states, texts, period=True),
        }


def problems(baseline: dict[str, Any], steered: dict[str, Any]) -> list[str]:
    """Reasons not to ship the vectors."""
    found = []
    onset, before = steered["onsets"], baseline["onsets"]
    if onset["speech_by_frame_1"] < max(0.7, before["speech_by_frame_1"] + 0.2):
        found.append(
            f"speech by frame 1 is {onset['speech_by_frame_1']:.0%} (baseline "
            f"{before['speech_by_frame_1']:.0%}); expected at least 70% and 20 points more"
        )
    # Relative to the model without vectors; the slack covers the noise of 180 trials.
    for section, label, slack in (
        ("tail_with_period", "with a period", 0.05),
        ("tail_without_period", "without a period", 0.1),
    ):
        for key, what in (
            ("tail_speech_rate", "speech after EOS"),
            ("early_eos_rate", "EOS on the last loud frame"),
        ):
            value, plain = steered[section][key], baseline[section][key]
            if value > plain + slack:
                found.append(f"{what} {label} in {value:.0%} of generations (baseline {plain:.0%})")
    return found
