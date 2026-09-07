from __future__ import annotations

from collections.abc import Iterable

import torch

from .records import WordEnd, WordStart
from .text import _SourceWord, _TextUnit

_SILENCE_RMS_THRESHOLD = torch.tensor(1e-3, dtype=torch.float32).item()
_NON_NEXT_ATTENTION_THRESHOLD = 0.001


def is_voiced(audio: torch.Tensor) -> bool:
    samples = audio.detach().to(dtype=torch.float32).reshape(-1)
    if not samples.numel():
        return False
    energy = torch.dot(samples, samples).item()
    threshold_energy = _SILENCE_RMS_THRESHOLD**2 * samples.numel()
    return energy > threshold_energy


class SelectedAttentionCapture:
    """Collect configured FlowLM heads and reduce them to text-unit scores."""

    def __init__(
        self,
        heads: Iterable[tuple[int, int]],
        text_start: int,
        text_end: int,
        token_to_unit: torch.Tensor,
    ):
        self.heads = tuple(heads)
        self.text_start = text_start
        self.text_end = text_end
        self.token_to_unit = token_to_unit
        heads_by_layer: dict[int, list[int]] = {}
        for layer, head in self.heads:
            heads_by_layer.setdefault(layer, []).append(head)
        self._heads_by_layer = {
            layer: tuple(layer_heads) for layer, layer_heads in heads_by_layer.items()
        }
        self._head_index_tensors: dict[tuple[int, torch.device], torch.Tensor] = {}
        self._prepared_mapping: torch.Tensor | None = None
        self._unit_score_sum: torch.Tensor | None = None
        self._recorded_head_count = 0

    def heads_for_layer(self, layer_index: int) -> tuple[int, ...]:
        return self._heads_by_layer.get(layer_index, ())

    def head_indices_for_layer(self, layer_index: int, device: torch.device) -> torch.Tensor:
        key = (layer_index, device)
        indices = self._head_index_tensors.get(key)
        if indices is None:
            indices = torch.tensor(
                self.heads_for_layer(layer_index), dtype=torch.long, device=device
            )
            self._head_index_tensors[key] = indices
        return indices

    def begin_frame(self) -> None:
        self._unit_score_sum = None
        self._recorded_head_count = 0

    def capture_attention(self, layer_index: int, query: torch.Tensor, keys: torch.Tensor) -> None:
        """Extract configured head scores from projected attention inputs."""
        if query.shape[2] != 1:
            return
        selected_heads = self.heads_for_layer(layer_index)
        if not selected_heads:
            return

        text_keys = keys[:, :, self.text_start : self.text_end]
        if len(selected_heads) == 1:
            head = selected_heads[0]
            selected_query = query[:, head, 0]
            selected_keys = text_keys[:, head]
            if query.shape[0] == 1:
                logits = torch.mv(selected_keys[0], selected_query[0])[None, None]
            else:
                logits = torch.bmm(selected_keys, selected_query.unsqueeze(-1)).squeeze(-1)[:, None]
        else:
            head_indices = self.head_indices_for_layer(layer_index, query.device)
            selected_query = query.index_select(1, head_indices)[:, :, 0]
            selected_keys = text_keys.index_select(1, head_indices)
            logits = torch.einsum("bhd,bhtd->bht", selected_query, selected_keys)

        logits = logits / query.shape[-1] ** 0.5
        attention = torch.softmax(logits.float(), dim=-1).to(query.dtype)
        self.record(layer_index, selected_heads, attention)

    def record(
        self, layer_index: int, head_indices: tuple[int, ...], attention: torch.Tensor
    ) -> None:
        if head_indices != self.heads_for_layer(layer_index):
            raise RuntimeError(f"Unexpected timestamp heads captured for layer {layer_index}")
        mapping = self._prepared_mapping
        if (
            mapping is None
            or mapping.device != attention.device
            or mapping.dtype != attention.dtype
        ):
            mapping = self.token_to_unit.to(device=attention.device, dtype=attention.dtype)
            self._prepared_mapping = mapping
        reduced = torch.matmul(attention, mapping).sum(dim=1)
        self._unit_score_sum = (
            reduced if self._unit_score_sum is None else self._unit_score_sum + reduced
        )
        self._recorded_head_count += len(head_indices)

    def finish_frame(self) -> torch.Tensor:
        if self._unit_score_sum is None or self._recorded_head_count != len(self.heads):
            raise RuntimeError("Timestamp attention was not captured for every configured head")
        return (self._unit_score_sum[0] / self._recorded_head_count).detach()


class WordAlignment:
    def __init__(self, units: tuple[_TextUnit, ...]):
        self.units = units
        self._word_unit_indices = [
            unit_index for unit_index, unit in enumerate(units) if unit.is_word
        ]
        self._next_word_position = 0
        self._open_word_position: int | None = None
        self._open_start: float | None = None

    def _word(self, position: int) -> _SourceWord:
        word = self.units[self._word_unit_indices[position]].word
        assert word is not None
        return word

    def _open_next(self, timestamp: float) -> WordStart:
        position = self._next_word_position
        word = self._word(position)
        self._open_word_position = position
        self._open_start = timestamp
        self._next_word_position = position + 1
        return WordStart(word.text, word.word_index, timestamp)

    def _close(self, timestamp: float) -> WordEnd:
        assert self._open_word_position is not None and self._open_start is not None
        word = self._word(self._open_word_position)
        event = WordEnd(word.text, word.word_index, self._open_start, timestamp)
        self._open_word_position = None
        self._open_start = None
        return event

    def _future_word_dominates(self, scores: torch.Tensor, current_unit: int) -> bool:
        current_score = scores[current_unit].item()
        next_unit = self._word_unit_indices[self._next_word_position]
        if scores[next_unit].item() > current_score:
            return True
        if current_score >= _NON_NEXT_ATTENTION_THRESHOLD:
            return False
        later_units = self._word_unit_indices[self._next_word_position + 1 :]
        return any(scores[unit].item() > current_score for unit in later_units)

    def process_frame(
        self, scores: torch.Tensor, voiced: bool, frame_start: float
    ) -> list[WordStart | WordEnd]:
        if scores.shape != (len(self.units),):
            raise ValueError(
                f"Expected {len(self.units)} timestamp unit scores, got {tuple(scores.shape)}"
            )
        events: list[WordStart | WordEnd] = []

        if not voiced:
            if self._open_word_position is None:
                return events
            current_unit = self._word_unit_indices[self._open_word_position]
            future = scores[current_unit + 1 :]
            should_close = (
                bool(future.numel()) and future.max().item() > scores[current_unit].item()
            )
            is_final_word = self._open_word_position == len(self._word_unit_indices) - 1
            has_later_punctuation = any(
                not unit.is_word and not unit.synthetic for unit in self.units[current_unit + 1 :]
            )
            if should_close or (is_final_word and not has_later_punctuation):
                events.append(self._close(frame_start))
            return events

        if self._open_word_position is None:
            if self._next_word_position >= len(self._word_unit_indices):
                return events
            events.append(self._open_next(frame_start))
            return events

        current_unit = self._word_unit_indices[self._open_word_position]
        if self._next_word_position >= len(self._word_unit_indices):
            return events
        if self._future_word_dominates(scores, current_unit):
            events.append(self._close(frame_start))
            events.append(self._open_next(frame_start))
        return events

    def finish(self, audio_end: float) -> list[WordEnd]:
        events: list[WordEnd] = []
        if self._open_word_position is not None:
            events.append(self._close(audio_end))
        self._next_word_position = len(self._word_unit_indices)
        return events
