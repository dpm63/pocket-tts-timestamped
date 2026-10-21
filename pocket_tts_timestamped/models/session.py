"""Frame-by-frame timestamped generation whose text can be replaced mid-utterance.

A `Session` holds one FlowLM state conditioned on a voice, one Mimi decoder and one word
alignment. `set_text` rewinds the FlowLM to the end of the voice prompt and has it read the new
text, then (optionally) the audio generated so far, in one pass. The decoder and the alignment
carry on, so the audio stays one continuous stream and word events keep counting.

Generation and decoding are separate steps so they can run on two threads: `decode` may run
while `generate_frame` or `set_text` runs on another thread, but `generate_frame` and
`set_text` must not run concurrently.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from pocket_tts_timestamped.data.audio_utils import fade_in_audio
from pocket_tts_timestamped.models.text_chunking import prepare_text_prompt
from pocket_tts_timestamped.modules.stateful_module import ModelState, increment_steps
from pocket_tts_timestamped.timestamps.alignment import (
    SelectedAttentionCapture,
    WordAlignment,
    is_voiced,
)
from pocket_tts_timestamped.timestamps.records import AudioChunk, TimestampEvent, WordEnd
from pocket_tts_timestamped.timestamps.text import (
    TimestampTextChunk,
    _iter_timestamp_text_chunks,
    offset_timestamp_text_chunk,
)

if TYPE_CHECKING:
    from pocket_tts_timestamped.models.tts_model import TTSModel

# FlowLM cache slots added at a time when generation outgrows the reserved capacity.
_FLOW_CACHE_GROWTH = 256


@dataclass(frozen=True, slots=True, eq=False)
class TextWindow:
    """Text prepared for `Session.set_text`, mapped back to the source text.

    Build one with `Session.prepare_text`.
    """

    chunk: TimestampTextChunk
    tokens: torch.Tensor
    # Frames the model recommends generating after EOS for this text.
    frames_after_eos: int


@dataclass(frozen=True, slots=True, eq=False)
class Frame:
    """One generated 80 ms frame, ready for `Session.decode`."""

    # Position among the session's kept frames (`Session.latents`), so in its audio.
    index: int
    # Frames generated since the last `set_text`.
    step: int
    # Normalized FlowLM latent, [1, 1, ldim].
    latent: torch.Tensor
    # The EOS head fired, at least `TTSModel._MIN_FRAMES_BEFORE_EOS` frames after the last
    # `set_text` (with or without history, as the input-streaming policy was tuned).
    eos: bool
    # Timestamp attention reduced to the window's text units; None without timestamp heads.
    scores: torch.Tensor | None
    # The text the frame was generated under.
    window: TextWindow


class Session:
    """Timestamped generation for one utterance; see the module docstring.

    Get one from `TTSModel.start_session`. Not thread-safe beyond the generate/decode split.

    With a quantized FlowLM (`quantize=True`), a replay is not equivalent to generating the
    same frames one by one: dynamic int8 scales activations per call, so every `set_text` with
    history shifts the state slightly (next latent off by about 0.1 after 8 frames).
    """

    def __init__(
        self,
        model: TTSModel,
        voice_state: ModelState,
        *,
        copy_state: bool = True,
        time_offset: float = 0.0,
        decoder_frames: int = 1000,
    ):
        """`decoder_frames` sizes Mimi caches that cannot grow; `copy_state=False` generates in
        `voice_state` itself. Both exist for `TTSModel`'s own generation."""
        self.model = model
        self._flow_state = copy.deepcopy(voice_state) if copy_state else voice_state
        # set_text rewinds to here: positions are restored, and cache entries past them are
        # overwritten before they are read.
        self._rewind_point = {
            name: {key: value.clone() for key, value in module_state.items() if key != "cache"}
            for name, module_state in self._flow_state.items()
        }
        self._prompt_end = model._flow_lm_current_end(self._flow_state)
        self._flow_end = self._prompt_end
        self._heads = tuple((head.layer, head.head) for head in model.config.timestamp_heads or ())
        self._bos = torch.full(
            (1, 1, model.flow_lm.ldim),
            float("NaN"),
            device=model.flow_lm.device,
            dtype=model.flow_lm.dtype,
        )
        self._mimi_steps = int(model.mimi.encoder_frame_rate / model.mimi.frame_rate)
        self._mimi_state, warm = model._new_mimi_state(decoder_frames * self._mimi_steps)
        # Generated latents the caller kept (see `discard`); `set_text` replays them.
        self.latents: list[torch.Tensor] = []
        self._window: TextWindow | None = None
        self._capture: SelectedAttentionCapture | None = None
        self._next_input: torch.Tensor | None = None
        self._step = 0
        # Decoder side.
        self._time_offset = time_offset
        self._decoded_samples = 0
        self._decoded_frames = 0
        # A fresh decoder clicks on its first samples; a warm one (leading-silence vectors) doesn't.
        self._fade_first_decode = not warm
        self._alignment: WordAlignment | None = None
        self._aligned_window: TextWindow | None = None

    @property
    def time(self) -> float:
        """End of the decoded audio, in seconds."""
        return self._time_offset + self._decoded_samples / self.model.sample_rate

    def prepare_text(
        self, text: str, start: int = 0, *, append_terminal_punctuation: bool | None = None
    ) -> TextWindow:
        """Prepare `text[start:]` for `set_text`. Word events report positions in `text`.

        `start` must be a word start, such as the `char_start` of a word event, or fall
        between words. `append_terminal_punctuation` defaults to the model's setting.
        """
        if 0 < start < len(text) and text[start - 1].isalnum() and text[start].isalnum():
            raise ValueError(f"start={start} falls inside a word")
        model = self.model
        source = text[start:]
        prepared, frames_guess = prepare_text_prompt(
            source,
            model.pad_with_spaces_for_short_inputs,
            model.remove_semicolons,
            model.append_terminal_punctuation
            if append_terminal_punctuation is None
            else append_terminal_punctuation,
            model.capitalize_first_letter,
            model.replace_characters,
        )
        chunk = next(
            _iter_timestamp_text_chunks(
                source, [prepared], model.flow_lm.conditioner.tokenizer, best_effort=True
            )
        )
        if start:
            chunk = offset_timestamp_text_chunk(chunk, text, start)
        recommended = model.model_recommended_frames_after_eos
        return TextWindow(
            chunk=chunk,
            tokens=model._prepare_timestamp_text_chunk(chunk),
            frames_after_eos=frames_guess + 2 if recommended is None else recommended,
        )

    @torch.no_grad
    def set_text(
        self, window: TextWindow, history_from: int | None = None, reserve_frames: int = 0
    ) -> None:
        """Generate under `window` from now on.

        With `history_from`, the FlowLM re-reads `latents[history_from:]` after the text, so
        generation continues the audio; the BOS before them is the plain model BOS. Without
        it (or with `history_from == len(latents)`, which keeps no audio), the next frame
        starts a new utterance from BOS, with the leading-silence vectors if the model has
        them. `reserve_frames` sizes the FlowLM cache for that many frames up front; it grows
        on demand either way.
        """
        if window.chunk.token_to_unit.shape[0] != window.tokens.shape[1]:
            raise RuntimeError("Timestamp token mapping does not match FlowLM tokenization")
        if history_from is not None and not 0 <= history_from <= len(self.latents):
            raise ValueError(
                f"history_from={history_from} is outside 0..{len(self.latents)} (kept frames)"
            )
        state = self._flow_state
        for name, saved in self._rewind_point.items():
            for key, value in saved.items():
                state[name][key].copy_(value)
        history = self.latents[history_from:] if history_from is not None else []
        if history:
            # BOS and every retained latent but the newest, which feeds the next frame.
            replay = torch.cat([self._bos, *history[:-1]], dim=1)
        else:
            replay = self._bos[:, :0]
        token_count = window.tokens.shape[1]
        self._reserve(self._prompt_end + token_count + replay.shape[1] + reserve_frames)
        # One pass over text and replay: each FlowLM call streams all of its weights, which
        # dominates the cost of short inputs. Its sampled latent is discarded.
        self.model._run_flow_lm_and_increment_step(
            model_state=state, text_tokens=window.tokens, backbone_input_latents=replay
        )
        self._flow_end = self._prompt_end + token_count + replay.shape[1]
        self._next_input = history[-1] if history else self._bos
        self._capture = (
            SelectedAttentionCapture(
                heads=self._heads,
                text_start=self._prompt_end,
                text_end=self._prompt_end + token_count,
                token_to_unit=window.chunk.token_to_unit,
            )
            if self._heads
            else None
        )
        self._window = window
        self._step = 0

    @torch.no_grad
    def generate_frame(self) -> Frame:
        """Generate the next frame with the FlowLM."""
        if self._window is None or self._next_input is None:
            raise RuntimeError("Call set_text before generate_frame (and after discard)")
        self._reserve(self._flow_end + 1)
        capture = self._capture
        if capture is not None:
            capture.begin_frame()
        latent, is_eos = self.model._run_flow_lm_and_increment_step(
            model_state=self._flow_state,
            backbone_input_latents=self._next_input,
            attention_capture=capture,
        )
        self._flow_end += 1
        frame = Frame(
            index=len(self.latents),
            step=self._step,
            latent=latent,
            eos=bool(is_eos.item()) and self._step >= self.model._MIN_FRAMES_BEFORE_EOS,
            scores=capture.finish_frame() if capture is not None else None,
            window=self._window,
        )
        self.latents.append(latent)
        self._next_input = latent
        self._step += 1
        return frame

    def discard(self, frame: Frame) -> None:
        """Drop the last generated frame from the history, e.g. an EOS frame that won't be
        played. It must not have been decoded. The next call must be `set_text`."""
        if not self.latents or frame.index != len(self.latents) - 1:
            raise ValueError("Only the last generated frame can be discarded")
        if frame.index < self._decoded_frames:
            raise ValueError("A decoded frame cannot be discarded")
        self.latents.pop()
        self._next_input = None

    @torch.no_grad
    def decode(self, frames: Sequence[Frame]) -> list[TimestampEvent]:
        """Decode the next frames, in generation order, in one Mimi call.

        Returns each frame's word events followed by its `AudioChunk`.
        """
        if not frames:
            return []
        indices = [frame.index for frame in frames]
        expected = list(range(self._decoded_frames, self._decoded_frames + len(frames)))
        if indices != expected:
            raise ValueError(
                f"Frames must be decoded once each, in order: expected {expected}, got {indices}"
            )
        self._decoded_frames += len(frames)
        model = self.model
        latents = torch.cat([frame.latent for frame in frames], dim=1)
        audio = model.mimi.decode_from_latent(
            latents * model.flow_lm.emb_std + model.flow_lm.emb_mean, self._mimi_state
        )
        if self._fade_first_decode:
            fade_in_audio(audio, model.sample_rate)
            self._fade_first_decode = False
        increment_steps(model.mimi, self._mimi_state, increment=self._mimi_steps * len(frames))
        events: list[TimestampEvent] = []
        for frame, frame_audio in zip(
            frames, audio[0, 0].split(model.mimi.frame_size), strict=True
        ):
            start = self.time
            self._decoded_samples += frame_audio.shape[-1]
            if frame.scores is not None:
                alignment = self._alignment_for(frame.window, start, events)
                events.extend(alignment.process_frame(frame.scores, is_voiced(frame_audio), start))
            events.append(AudioChunk(audio=frame_audio, start_time=start, end_time=self.time))
        return events

    def finish(self) -> list[WordEnd]:
        """Close the word being spoken at the end of the decoded audio. No word opens after
        this, so call it once the utterance is over."""
        return [] if self._alignment is None else self._alignment.finish(self.time)

    def _alignment_for(
        self, window: TextWindow, at: float, events: list[TimestampEvent]
    ) -> WordAlignment:
        """The alignment for `window`, taking over progress when the window changes. A word
        left open that the new window lacks is closed at `at`, into `events`."""
        if window is not self._aligned_window:
            alignment = WordAlignment(window.chunk.units)
            if self._alignment is not None:
                events.extend(alignment.continue_from(self._alignment, at))
            self._alignment = alignment
            self._aligned_window = window
        assert self._alignment is not None
        return self._alignment

    def _reserve(self, slots: int) -> None:
        """Make the FlowLM caches (all the same length) hold at least `slots` positions."""
        capacity = next(
            (
                module_state["cache"].shape[2]
                for module_state in self._flow_state.values()
                if "cache" in module_state
            ),
            None,
        )
        if capacity is not None and capacity < slots:
            self.model._expand_kv_cache(self._flow_state, max(slots, capacity + _FLOW_CACHE_GROWTH))
