"""Whisper on whole generations: whether the first word is spoken, and the word error rate.

Onset metrics alone can't tell a quick start from a clipped or garbled first word: on
english_drifting_26-09, a setting with the best onsets and no abrupt starts said "lumi" for
"Jimmy" (bos_shortcut handoff, section 3). Whisper catches that.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

WHISPER_MODEL = "large-v3-turbo"  # what the research scored with
WHISPER_SAMPLE_RATE = 16000
LEAD_PAD_S = 0.3  # Whisper misses words that start at sample 0
LANGUAGE_CODES = {
    "english": "en",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "italian": "it",
    "portuguese": "pt",
    "dutch": "nl",
}


def load_whisper(threads: int) -> Any:  # noqa: ANN401
    from faster_whisper import WhisperModel  # ty: ignore[unresolved-import]  -- requirements.txt

    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=threads)


def words_of(text: str) -> list[str]:
    # Letters of any script (same as [a-z']+ on ASCII English).
    return re.findall(r"(?:[^\W\d_]|')+", text.lower().replace("’", "'"))


def _edit_distance(a: list[str], b: list[str]) -> int:
    row = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        previous, row[0] = row[0], i
        for j in range(1, len(b) + 1):
            previous, row[j] = (
                row[j],
                min(row[j] + 1, row[j - 1] + 1, previous + (a[i - 1] != b[j - 1])),
            )
    return row[len(b)]


def transcribe_clip(
    whisper: Any,  # noqa: ANN401
    pcm: NDArray[np.float32],
    text: str,
    sample_rate: int,
    language: str,
) -> dict[str, Any]:
    """What Whisper hears in one clip of `text`: its words, first word right, WER."""
    from scipy.signal import resample_poly

    padded = np.concatenate([np.zeros(int(LEAD_PAD_S * sample_rate), np.float32), pcm])
    audio = resample_poly(padded, WHISPER_SAMPLE_RATE, sample_rate).astype(np.float32)
    segments, _ = whisper.transcribe(audio, language=LANGUAGE_CODES[language])
    hypothesis = words_of(" ".join(segment.text for segment in segments))
    reference = words_of(text)
    return {
        "heard": " ".join(hypothesis),
        "first_word_right": hypothesis[:1] == reference[:1],
        "wer": _edit_distance(reference, hypothesis) / max(1, len(reference)),
    }


def score(
    whisper: Any,  # noqa: ANN401
    clips: list[tuple[NDArray[np.float32], str]],
    sample_rate: int,
    language: str,
) -> dict[str, float]:
    """First word right and mean WER of (pcm, text) clips."""
    heard = [transcribe_clip(whisper, pcm, text, sample_rate, language) for pcm, text in clips]
    first = [clip["first_word_right"] for clip in heard]
    errors = [clip["wer"] for clip in heard]
    return {
        "trials": len(clips),
        "first_word_right": round(float(np.mean(first)), 3) if clips else float("nan"),
        "wer": round(float(np.mean(errors)), 3) if clips else float("nan"),
    }
