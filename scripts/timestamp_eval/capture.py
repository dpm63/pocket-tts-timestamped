"""Generate once; retain all head-to-unit scores and individual predictions."""

from __future__ import annotations

import time
from collections.abc import Generator as EventGenerator
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io.wavfile
import torch
from numpy.typing import NDArray

from pocket_tts_timestamped import TTSModel
from pocket_tts_timestamped.models import tts_model as model_module
from pocket_tts_timestamped.models.tts_model import ModelState
from pocket_tts_timestamped.timestamps import (
    SelectedAttentionCapture,
    TimestampEvent,
    TimestampTextChunk,
    is_voiced,
)
from pocket_tts_timestamped.utils.config import TimestampHeadConfig
from pocket_tts_timestamped.utils.utils import download_if_necessary
from scripts.timestamp_eval.alignment import mixtures, predict
from scripts.timestamp_eval.common import file_hash, lexical_words, read, write


class AllHeadCapture(SelectedAttentionCapture):
    def __init__(
        self,
        heads: Iterable[tuple[int, int]],
        text_start: int,
        text_end: int,
        token_to_unit: torch.Tensor,
    ):
        super().__init__(heads, text_start, text_end, token_to_unit)
        self.frames: dict[int, list[torch.Tensor]] = {}

    def record(
        self, layer_index: int, head_indices: tuple[int, ...], attention: torch.Tensor
    ) -> None:
        reduced = torch.matmul(attention[0], self.token_to_unit.to(attention)).detach().cpu()
        self.frames.setdefault(layer_index, []).append(reduced)
        super().record(layer_index, head_indices, attention)

    def finish(self) -> NDArray[np.float32]:
        counts = [len(frames) for frames in self.frames.values()]
        if not counts or min(counts) == 0 or len(set(counts)) != 1:
            raise RuntimeError(f"Incomplete all-head capture: {counts}")
        return torch.cat(
            [torch.stack(self.frames[layer], dim=1) for layer in sorted(self.frames)]
        ).numpy()


class Generator:
    def __init__(self, item: dict[str, Any]):
        self.item = item
        self.model = TTSModel.load_model(language=item["id"]).eval()
        effective = (
            self.model.config.weights_path
            if self.model.has_voice_cloning
            else self.model.config.weights_path_without_voice_cloning
        )
        effective = effective or self.model.config.flow_lm.weights_path
        if (
            effective is None
            or file_hash(download_if_necessary(effective)) != item["checkpoint_sha256"]
        ):
            raise RuntimeError("Loaded checkpoint does not match planned content hash")
        shape = self.model.config.flow_lm.transformer
        self.heads = [
            [layer, head] for layer in range(shape.num_layers) for head in range(shape.num_heads)
        ]
        self.model.config.timestamp_heads = [
            TimestampHeadConfig(layer=layer, head=head) for layer, head in self.heads
        ]
        self.states: dict[str, Any] = {}

    def generate(self, sample: dict[str, Any], directory: Path) -> dict[str, Any]:
        marker = directory / "sample.json"
        if marker.exists():
            metadata = read(marker)
            if (
                metadata["checkpoint_sha256"] != self.item["checkpoint_sha256"]
                or metadata["seed"] != sample["seed"]
            ):
                raise RuntimeError("Incompatible partial generation cache")
            return metadata
        model = self.model
        voice = sample["voice"]
        if voice not in self.states:
            self.states[voice] = model.get_state_for_audio_prompt(voice)
        captures: list[AllHeadCapture] = []
        chunks: list[TimestampTextChunk] = []
        durations: list[float] = []
        original_capture = model_module.SelectedAttentionCapture
        original_prepare = model._prepare_timestamp_text_chunk
        original_short_text = model._generate_audio_with_timestamps_short_text

        def factory(
            heads: Iterable[tuple[int, int]],
            text_start: int,
            text_end: int,
            token_to_unit: torch.Tensor,
        ) -> AllHeadCapture:
            capture = AllHeadCapture(heads, text_start, text_end, token_to_unit)
            captures.append(capture)
            return capture

        def prepare(chunk: TimestampTextChunk) -> torch.Tensor:
            chunks.append(chunk)
            return original_prepare(chunk)

        def short_text(
            model_state: ModelState,
            timestamp_chunk: TimestampTextChunk,
            frames_after_eos: int,
            copy_state: bool,
            time_offset: float,
        ) -> EventGenerator[TimestampEvent, None, float]:
            end = yield from original_short_text(
                model_state, timestamp_chunk, frames_after_eos, copy_state, time_offset
            )
            durations.append(end - time_offset)
            return end

        model_module.SelectedAttentionCapture = factory  # ty: ignore[invalid-assignment]
        setattr(model, "_prepare_timestamp_text_chunk", prepare)
        setattr(model, "_generate_audio_with_timestamps_short_text", short_text)
        torch.manual_seed(sample["seed"])
        started = time.perf_counter()
        try:
            result = model.generate_audio_with_timestamps(
                self.states[voice], sample["text"], copy_state=True
            )
        finally:
            model_module.SelectedAttentionCapture = original_capture
            setattr(model, "_prepare_timestamp_text_chunk", original_prepare)
            setattr(model, "_generate_audio_with_timestamps_short_text", original_short_text)
        audio = result.audio.detach().cpu().numpy()
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("Invalid generated audio")
        words = lexical_words(sample["text"])
        directory.mkdir(parents=True, exist_ok=True)
        scipy.io.wavfile.write(directory / "audio.wav", model.sample_rate, audio)
        arrays: dict[str, Any] = {}
        chunk_meta = []
        offset_frames = 0
        frame_rate = float(model.config.mimi.frame_rate)
        samples_per_frame = round(model.sample_rate / frame_rate)
        baseline = self.item["baseline"]
        baseline_indices = [self.heads.index(head) for head in baseline]
        groups = [[i] for i in range(len(self.heads))] + (
            [baseline_indices] if baseline_indices else []
        )
        bounds = np.full((len(groups), len(words), 2), np.nan)
        for index, (capture, chunk, duration) in enumerate(
            zip(captures, chunks, durations, strict=True)
        ):
            scores = capture.finish()
            # The terminal EOS step captures attention but does not emit a latent.
            # Trim each chunk to its emitted audio, rather than borrowing a frame
            # from the next chunk in a multi-chunk sample.
            frames = round(duration * frame_rate)
            if not frames <= scores.shape[1] <= frames + 1:
                raise RuntimeError("Captured frame count differs from emitted chunk audio")
            scores = scores[:, :frames]
            voiced = np.asarray(
                [
                    is_voiced(
                        torch.from_numpy(
                            audio[
                                (offset_frames + f) * samples_per_frame : (offset_frames + f + 1)
                                * samples_per_frame
                            ]
                        )
                    )
                    for f in range(frames)
                ],
                dtype=np.bool_,
            )
            unit_indices = np.asarray(
                [i for i, u in enumerate(chunk.units) if u.is_word], dtype=np.int64
            )
            word_indices = []
            for unit in unit_indices:
                word = chunk.units[unit].word
                assert word is not None
                word_indices.append(word.word_index)
            punctuation = np.asarray(
                [
                    any(not u.is_word and not u.synthetic for u in chunk.units[int(i) + 1 :])
                    for i in unit_indices
                ],
                dtype=np.bool_,
            )
            predicted = (
                predict(
                    mixtures(scores, groups),
                    voiced,
                    unit_indices,
                    punctuation,
                    frame_rate,
                    duration,
                )
                + offset_frames / frame_rate
            )
            bounds[:, word_indices] = predicted
            arrays.update(
                {
                    f"scores_{index}": scores,
                    f"voiced_{index}": voiced,
                    f"word_units_{index}": unit_indices,
                    f"punctuation_{index}": punctuation,
                }
            )
            chunk_meta.append(
                {
                    "word_indices": word_indices,
                    "offset_seconds": offset_frames / frame_rate,
                    "duration": duration,
                }
            )
            offset_frames += frames
        if offset_frames * samples_per_frame != len(audio):
            raise ValueError("Capture frames do not cover generated audio exactly")
        np.savez_compressed(directory / "units.npz", **arrays)
        np.savez_compressed(directory / "singles.npz", bounds=bounds)
        metadata = {
            **sample,
            "checkpoint_sha256": self.item["checkpoint_sha256"],
            "heads": self.heads,
            "baseline": baseline,
            "words": words,
            "chunks": chunk_meta,
            "frame_rate": frame_rate,
            "sample_rate": model.sample_rate,
            "audio_samples": len(audio),
            "audio_sha256": file_hash(directory / "audio.wav"),
            "generation_seconds": time.perf_counter() - started,
        }
        write(marker, metadata)
        return metadata


def replay(directory: Path, groups: list[list[list[int]]]) -> NDArray[np.float64]:
    metadata = read(directory / "sample.json")
    indices = [[metadata["heads"].index(head) for head in group] for group in groups]
    result = np.full((len(groups), len(metadata["words"]), 2), np.nan)
    with np.load(directory / "units.npz") as data:
        for index, chunk in enumerate(metadata["chunks"]):
            values = mixtures(data[f"scores_{index}"], indices)
            predicted = (
                predict(
                    values,
                    data[f"voiced_{index}"],
                    data[f"word_units_{index}"],
                    data[f"punctuation_{index}"],
                    metadata["frame_rate"],
                    chunk["duration"],
                )
                + chunk["offset_seconds"]
            )
            result[:, chunk["word_indices"]] = predicted
    return result
