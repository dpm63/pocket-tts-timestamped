"""Speech onset by loudness, per 80 ms frame (the rules the vectors were developed with)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FRAME_SAMPLES = 1920
VOICED_DB = -60.0  # a frame above this is voiced
MIN_RUN = 3  # voiced frames in a row that make a segment
SPEECH_DB = -35.0  # speech proper, past breaths and room noise
SPEECH_WINDOW = 4  # speech_onset: a loud frame with another among the next 3
PAUSE_DB = -55.0  # a frame at or below this ends a voiced segment
MAX_BREATH = 10  # the longest segment that can be skipped as a breath
ABRUPT_DB = -45.0  # the first 30 ms of an onset louder than this = an abrupt start
HEAD_MS = 30


def frame_db(pcm: NDArray[Any]) -> NDArray[np.float64]:
    count = len(pcm) // FRAME_SAMPLES
    frames = pcm[: count * FRAME_SAMPLES].reshape(count, FRAME_SAMPLES).astype(np.float64)
    return 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-12)


def sustained(voiced: NDArray[np.bool_], min_run: int = MIN_RUN) -> int:
    """Start of the first `min_run` voiced frames in a row; -1 if none."""
    for index in range(len(voiced) - min_run + 1):
        if voiced[index : index + min_run].all():
            return index
    return -1


def speech_start(db: NDArray[np.float64]) -> int:
    """The training target: the first sustained voiced segment that is not a breath; -1 if
    none. A breath never reaches SPEECH_DB, lasts at most MAX_BREATH frames and is followed
    by a pause. The onset keeps the soft ramp-up frames of speech."""
    voiced, start = db > VOICED_DB, 0
    while True:
        found = sustained(voiced[start:])
        if found < 0:
            return -1
        onset = start + found
        end = onset + 1
        while end < len(db) and db[end] > PAUSE_DB:
            end += 1
        quiet = bool((db[onset:end] <= SPEECH_DB).all())
        breath = end < len(db) and end - onset <= MAX_BREATH and quiet
        if not breath:
            return onset
        start = end


def speech_onset(db: NDArray[np.float64]) -> int:
    """For evaluation: the first frame above SPEECH_DB with another within the next 3 frames
    (quiet or choppy speech still counts from its first frame); -1 if none."""
    loud = db > SPEECH_DB
    for index in np.flatnonzero(loud):
        if loud[index + 1 : index + SPEECH_WINDOW].any():
            return int(index)
    return -1


VAD_SAMPLE_RATE, VAD_WINDOW = 16000, 512  # Silero VAD: 32 ms windows at 16 kHz
VAD_THRESHOLD = 0.5
VAD_MIN_WINDOWS = 3  # speech must hold this many windows in a row (96 ms)
# The VAD misses speech that ends the clip, so onset trials run this much past loud speech.
VAD_TAIL_FRAMES = 4


def vad_onset(pcm: NDArray[Any], sample_rate: int) -> int:
    """For evaluation: the 80 ms frame where the VAD's first speech run starts; -1 if none.

    Unlike the loudness rules it counts quiet speech from its first window and ignores breaths
    and noise, whatever their level; loudness made quiet voices look late (bos_shortcut
    handoff, "Measure speech onset with the VAD"). It lags 1-3 frames on unvoiced starts
    ("Stick", "Say"). The model is the Silero VAD that faster-whisper bundles.
    """
    from faster_whisper.vad import (
        get_vad_model,  # ty: ignore[unresolved-import]  -- requirements.txt
    )
    from scipy.signal import resample_poly

    x = resample_poly(pcm.astype(np.float32), VAD_SAMPLE_RATE, sample_rate).astype(np.float32)
    x = np.pad(x, (0, -len(x) % VAD_WINDOW))
    speech = get_vad_model()(x[None])[0] >= VAD_THRESHOLD
    for index in range(len(speech) - VAD_MIN_WINDOWS + 1):
        if speech[index : index + VAD_MIN_WINDOWS].all():
            return int(index * VAD_WINDOW * sample_rate // VAD_SAMPLE_RATE // FRAME_SAMPLES)
    return -1


def head_db(pcm: NDArray[Any], frame: int, sample_rate: int) -> float:
    """RMS dB of the first HEAD_MS of `frame`: natural onsets start near-silent and ramp up."""
    start = frame * FRAME_SAMPLES
    head = pcm[start : start + HEAD_MS * sample_rate // 1000].astype(np.float64)
    return float(20 * np.log10(np.sqrt(np.mean(head**2)) + 1e-9)) if len(head) else float("nan")
