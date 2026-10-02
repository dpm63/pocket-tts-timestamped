"""Compiled replay of the production WordAlignment state machine."""

from __future__ import annotations

from importlib import import_module

import numpy as np
from numpy.typing import NDArray

njit = import_module("numba").njit


@njit(cache=True)
def predict(
    values: NDArray[np.float32],
    voiced: NDArray[np.bool_],
    word_units: NDArray[np.int64],
    later_punctuation: NDArray[np.bool_],
    frame_rate: float,
    duration: float,
) -> NDArray[np.float64]:
    groups, frames, units = values.shape
    words = len(word_units)
    result = np.full((groups, words, 2), np.nan)
    for group in range(groups):
        next_word = 0
        open_word = -1
        open_start = 0.0
        for frame in range(min(frames, len(voiced))):
            stamp = frame / frame_rate
            close = False
            advance = False
            if not voiced[frame]:
                if open_word < 0:
                    continue
                current = word_units[open_word]
                for unit in range(current + 1, units):
                    if values[group, frame, unit] > values[group, frame, current]:
                        close = True
                        break
                if open_word == words - 1 and not later_punctuation[open_word]:
                    close = True
            elif open_word < 0:
                if next_word >= words:
                    continue
                open_word = next_word
                next_word += 1
                open_start = stamp
                continue
            else:
                if next_word >= words:
                    continue
                current_score = values[group, frame, word_units[open_word]]
                if values[group, frame, word_units[next_word]] > current_score:
                    close = True
                elif current_score < 0.001:
                    for word in range(next_word + 1, words):
                        if values[group, frame, word_units[word]] > current_score:
                            close = True
                            break
                advance = close
            if close:
                result[group, open_word, 0] = open_start
                result[group, open_word, 1] = stamp
                open_word = -1
                if advance:
                    open_word = next_word
                    next_word += 1
                    open_start = stamp
        if open_word >= 0:
            result[group, open_word, 0] = open_start
            result[group, open_word, 1] = duration
    return result


def mixtures(scores: NDArray[np.float32], groups: list[list[int]]) -> NDArray[np.float32]:
    weights = np.zeros((len(groups), scores.shape[0]), dtype=np.float32)
    for index, group in enumerate(groups):
        weights[index, group] = 1 / len(group)
    return (weights @ scores.reshape(scores.shape[0], -1)).reshape(
        len(groups), scores.shape[1], scores.shape[2]
    )


def metrics(
    bounds: NDArray[np.float64], expected: int, reference: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    present = np.asarray(np.isfinite(bounds).all(axis=2), dtype=np.bool_)
    result = np.zeros((bounds.shape[0], 4), dtype=np.float64)
    result[:, 0] = expected
    result[:, 1] = present.sum(1)
    if reference is not None:
        if reference.shape != (expected, 2):
            raise ValueError("Reference word count differs from input")
        result[:, 2:] = np.where(
            present[:, :, None], abs(bounds - reference[None, :, :]) * 1000, 0
        ).sum(1)
    return result
