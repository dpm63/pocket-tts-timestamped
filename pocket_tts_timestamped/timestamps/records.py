from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class AudioChunk:
    audio: torch.Tensor
    start_time: float
    end_time: float


# `char_start` / `char_end` locate each word in the generated text: `text[char_start:char_end]`.


@dataclass(frozen=True, slots=True)
class WordStart:
    word: str
    word_index: int
    start_time: float
    char_start: int | None = None
    char_end: int | None = None


@dataclass(frozen=True, slots=True)
class WordEnd:
    word: str
    word_index: int
    start_time: float
    end_time: float
    char_start: int | None = None
    char_end: int | None = None


@dataclass(frozen=True, slots=True)
class WordTimestamp:
    word: str
    word_index: int
    start_time: float
    end_time: float
    char_start: int | None = None
    char_end: int | None = None


@dataclass(frozen=True, slots=True)
class TimestampedAudio:
    audio: torch.Tensor
    words: tuple[WordTimestamp, ...]


TimestampEvent = AudioChunk | WordStart | WordEnd
