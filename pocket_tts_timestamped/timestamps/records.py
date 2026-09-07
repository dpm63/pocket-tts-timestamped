from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class AudioChunk:
    audio: torch.Tensor
    start_time: float
    end_time: float


@dataclass(frozen=True, slots=True)
class WordStart:
    word: str
    word_index: int
    start_time: float


@dataclass(frozen=True, slots=True)
class WordEnd:
    word: str
    word_index: int
    start_time: float
    end_time: float


@dataclass(frozen=True, slots=True)
class WordTimestamp:
    word: str
    word_index: int
    start_time: float
    end_time: float


@dataclass(frozen=True, slots=True)
class TimestampedAudio:
    audio: torch.Tensor
    words: tuple[WordTimestamp, ...]


TimestampEvent = AudioChunk | WordStart | WordEnd
