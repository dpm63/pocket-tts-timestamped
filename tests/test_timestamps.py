# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, ANN204
import copy
import queue
import threading
import unicodedata
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import sentencepiece
import torch
from torch import nn

import pocket_tts_timestamped.timestamps.alignment as timestamp_alignment
from pocket_tts_timestamped.models.tts_model import TTSModel
from pocket_tts_timestamped.modules.attention import StreamingMultiheadAttention
from pocket_tts_timestamped.modules.rope import RotaryEmbedding
from pocket_tts_timestamped.timestamps import (
    AudioChunk,
    SelectedAttentionCapture,
    TimestampedAudio,
    TimestampTextChunk,
    WordAlignment,
    WordEnd,
    WordStart,
    WordTimestamp,
    _SourceWord,
    _TextUnit,
    build_timestamp_text_chunks,
    is_voiced,
)
from pocket_tts_timestamped.utils.config import CONFIGS_DIR, Config, load_config
from pocket_tts_timestamped.utils.utils import download_if_necessary


class _SentencePiece021:
    def encode(self, text, out_type):
        assert out_type == "immutable_proto"
        pieces = [
            SimpleNamespace(
                id=index, piece=character, surface=character, begin=index, end=index + 1
            )
            for index, character in enumerate(text)
        ]
        return SimpleNamespace(pieces=pieces)


class _SentencePiece022:
    def encode(self, text, return_type, return_bytes):
        assert return_type == "offset_mapping"
        assert return_bytes
        offsets = []
        byte_offset = 0
        for character in text:
            byte_end = byte_offset + len(character.encode("utf-8"))
            offsets.append((byte_offset, byte_end))
            byte_offset = byte_end
        return {"ids": list(range(len(offsets))), "pieces": list(text), "offsets": offsets}


def _units(*values):
    units = []
    word_index = 0
    position = 0
    for value in values:
        if value.startswith("P:"):
            text = value[2:]
            units.append(_TextUnit(text, position, position + len(text), None))
        else:
            source = _SourceWord(value, word_index, position, position + len(value))
            units.append(_TextUnit(value, position, position + len(value), source))
            word_index += 1
        position += len(value)
    return tuple(units)


def test_text_chunks_preserve_original_words_and_map_punctuation():
    chunks = build_timestamp_text_chunks(
        "hello, blue-green world!", ["Hello, blue-green world!"], _SentencePiece021()
    )
    assert len(chunks) == 1
    chunk = chunks[0]
    assert [word.text for word in chunk.words] == ["hello", "blue-green", "world"]
    assert [unit.text for unit in chunk.units if not unit.is_word] == [",", "!"]
    assert torch.all(chunk.token_to_unit.sum(dim=1) <= 1)


def test_text_chunks_compare_sentencepiece_byte_offsets_with_utf8_unit_spans():
    chunk = build_timestamp_text_chunks("ação de", ["ação de"], _SentencePiece022())[0]

    assert torch.equal(chunk.token_to_unit[:4], torch.tensor([[1.0, 0.0]] * 4))
    assert torch.equal(chunk.token_to_unit[4], torch.tensor([0.0, 0.0]))
    assert torch.equal(chunk.token_to_unit[5:], torch.tensor([[0.0, 1.0]] * 2))


def test_text_chunks_compare_sentencepiece_character_offsets_with_character_unit_spans():
    chunk = build_timestamp_text_chunks("ação de", ["ação de"], _SentencePiece021())[0]

    assert torch.equal(chunk.token_to_unit[:4], torch.tensor([[1.0, 0.0]] * 4))
    assert torch.equal(chunk.token_to_unit[4], torch.tensor([0.0, 0.0]))
    assert torch.equal(chunk.token_to_unit[5:], torch.tensor([[0.0, 1.0]] * 2))


def test_appended_terminal_punctuation_is_marked_synthetic():
    chunk = build_timestamp_text_chunks("hello world", ["Hello world."], _SentencePiece021())[0]
    punctuation = [unit for unit in chunk.units if not unit.is_word]
    assert len(punctuation) == 1
    assert punctuation[0].synthetic

    alignment = WordAlignment(chunk.units)
    alignment.process_frame(torch.tensor([0.1, 0.9, 0.0]), True, 0.0)
    alignment.process_frame(torch.tensor([0.1, 0.9, 0.0]), True, 0.08)
    assert alignment.process_frame(torch.tensor([0.1, 0.9, 0.0]), False, 0.16) == [
        WordEnd("world", 1, 0.08, 0.16)
    ]


def test_alignment_transitions_to_next_word_and_hard_finishes():
    alignment = WordAlignment(_units("one", "two"))
    first = alignment.process_frame(torch.tensor([0.8, 0.2]), voiced=True, frame_start=0.0)
    transition = alignment.process_frame(torch.tensor([0.2, 0.8]), voiced=True, frame_start=0.08)
    final = alignment.finish(0.16)

    assert first == [WordStart("one", 0, 0.0)]
    assert transition == [WordEnd("one", 0, 0.0, 0.08), WordStart("two", 1, 0.08)]
    assert final == [WordEnd("two", 1, 0.08, 0.16)]


@pytest.fixture
def non_next_attention_gate(monkeypatch):
    threshold = 0.1
    monkeypatch.setattr(timestamp_alignment, "_NON_NEXT_ATTENTION_THRESHOLD", threshold)
    return threshold


def test_later_word_evidence_below_gate_advances_one_word(non_next_attention_gate):
    alignment = WordAlignment(_units("one", "two", "three"))
    alignment.process_frame(torch.tensor([0.9, 0.05, 0.05]), True, 0.0)
    events = alignment.process_frame(
        torch.tensor([non_next_attention_gate / 2, 0.01, 0.9]), True, 0.08
    )

    assert events == [WordEnd("one", 0, 0.0, 0.08), WordStart("two", 1, 0.08)]


def test_non_next_evidence_above_gate_is_blocked(non_next_attention_gate):
    alignment = WordAlignment(_units("one", "two", "three"))
    alignment.process_frame(torch.tensor([0.9, 0.05, 0.05]), True, 0.0)

    assert (
        alignment.process_frame(torch.tensor([non_next_attention_gate * 2, 0.01, 0.9]), True, 0.08)
        == []
    )


def test_non_next_evidence_uses_no_distance_margin(non_next_attention_gate):
    alignment = WordAlignment(_units("one", "two", "three", "four"))
    alignment.process_frame(torch.tensor([0.9, 0.05, 0.03, 0.02]), True, 0.0)
    current_score = non_next_attention_gate / 2
    assert alignment.process_frame(
        torch.tensor([current_score, 0.01, 0.01, current_score * 1.01]), True, 0.08
    )


def test_gate_never_blocks_next_word_evidence():
    alignment = WordAlignment(_units("one", "two", "three"))
    alignment.process_frame(torch.tensor([0.9, 0.05, 0.05]), True, 0.0)

    assert alignment.process_frame(torch.tensor([0.5, 0.51, 0.9]), True, 0.08)


def test_silence_closes_for_future_punctuation_but_never_opens():
    alignment = WordAlignment(_units("one", "P:."))
    alignment.process_frame(torch.tensor([0.8, 0.2]), True, 0.0)
    events = alignment.process_frame(torch.tensor([0.2, 0.8]), False, 0.08)
    assert events == [WordEnd("one", 0, 0.0, 0.08)]
    assert alignment.process_frame(torch.tensor([0.1, 0.9]), False, 0.16) == []


def test_unpunctuated_final_word_closes_on_first_silence():
    alignment = WordAlignment(_units("one"))
    alignment.process_frame(torch.tensor([1.0]), True, 0.0)
    assert alignment.process_frame(torch.tensor([1.0]), False, 0.08) == [
        WordEnd("one", 0, 0.0, 0.08)
    ]


@pytest.mark.parametrize(("dbfs", "expected"), [(-59.0, True), (-61.0, False)])
def test_rms_silence_gate_uses_configured_threshold(monkeypatch, dbfs, expected):
    monkeypatch.setattr(timestamp_alignment, "_SILENCE_RMS_THRESHOLD", 10 ** (-60.0 / 20))
    amplitude = 10 ** (dbfs / 20)
    assert is_voiced(torch.full((100,), amplitude)) is expected


def test_empty_audio_is_silent():
    assert not is_voiced(torch.empty(0))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_silence_detection_supports_non_contiguous_float_inputs(monkeypatch, dtype):
    monkeypatch.setattr(timestamp_alignment, "_SILENCE_RMS_THRESHOLD", 0.1)
    samples = torch.full((200,), 0.2, dtype=dtype)[::2]
    assert not samples.is_contiguous()
    assert is_voiced(samples)


def test_timestamp_chunk_reuses_prepared_tokens_and_keeps_manual_fallback():
    model = object.__new__(TTSModel)
    torch.nn.Module.__init__(model)
    prepared_calls = []
    model.flow_lm = SimpleNamespace(  # ty: ignore[invalid-assignment]
        device="cpu",
        conditioner=SimpleNamespace(
            prepare=lambda text: prepared_calls.append(text)
            or torch.tensor([[9]], dtype=torch.long)
        ),
    )

    retained = TimestampTextChunk(
        "one", _units("one"), torch.ones(1, 1), torch.tensor([[3]], dtype=torch.long)
    )
    prepared = model._prepare_timestamp_text_chunk(retained)
    assert prepared.tolist() == [[3]]
    assert prepared_calls == []

    manual = TimestampTextChunk("one", _units("one"), torch.ones(1, 1))
    prepared = model._prepare_timestamp_text_chunk(manual)
    assert prepared.tolist() == [[9]]
    assert prepared_calls == ["one"]


def test_closing_timestamp_generation_joins_workers_and_stops_state_mutation():
    model = object.__new__(TTSModel)
    torch.nn.Module.__init__(model)
    model.flow_lm = SimpleNamespace(  # ty: ignore[invalid-assignment]
        conditioner=SimpleNamespace(prepare=lambda _text: torch.zeros((1, 1), dtype=torch.long))
    )
    model.mimi = SimpleNamespace(encoder_frame_rate=1, frame_rate=1)  # ty: ignore[invalid-assignment]
    model.config = SimpleNamespace(  # ty: ignore[invalid-assignment]
        timestamp_heads=[SimpleNamespace(layer=0, head=0)]
    )
    model._estimate_max_gen_len = lambda _token_count: 1  # ty: ignore[invalid-assignment]
    model._flow_lm_current_end = lambda _model_state: 0  # ty: ignore[invalid-assignment]
    model_state = {"generation_steps": 0, "decoder_steps": 0}
    threads: list[threading.Thread] = []

    def decoder_worker(latents_queue, result_queue, *_args):
        threads.append(threading.current_thread())
        cancel_event = _args[-1]
        while not cancel_event.is_set():
            item = latents_queue.get()
            if item is None:
                return
            model_state["decoder_steps"] += 1
            result_queue.put(("event", AudioChunk(torch.ones(1), 0.0, 0.08)))

    def generate(**kwargs):
        cancel_event = kwargs["cancel_event"]
        latents_queue = kwargs["latents_queue"]

        def run():
            while not cancel_event.is_set():
                model_state["generation_steps"] += 1
                latents_queue.put((torch.empty(0), torch.ones(1)))
                cancel_event.wait(0.001)

        thread = threading.Thread(target=run)
        threads.append(thread)
        thread.start()
        return thread

    model._decode_timestamped_audio_worker = decoder_worker  # ty: ignore[invalid-assignment]
    model._generate = generate  # ty: ignore[invalid-assignment]
    timestamp_chunk = TimestampTextChunk("one", _units("one"), torch.ones(1, 1))
    generator = TTSModel._generate_audio_with_timestamps_short_text(
        model,
        model_state=model_state,  # ty: ignore[invalid-argument-type]
        timestamp_chunk=timestamp_chunk,
        frames_after_eos=0,
        copy_state=False,
        time_offset=0.0,
    )

    assert isinstance(next(generator), AudioChunk)
    generator.close()
    state_after_close = model_state.copy()

    assert all(not thread.is_alive() for thread in threads)
    assert model_state == state_after_close


def test_timestamp_decoder_passes_time_major_latent_to_mimi():
    class RecordingMimi(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoded_latents = []

        def quantizer(self, _latent):
            raise AssertionError("decode_from_latent owns quantization")

        def decode_from_latent(self, latent, _state):
            self.decoded_latents.append(latent)
            return torch.zeros(1, 1, 4)

    model = object.__new__(TTSModel)
    nn.Module.__init__(model)
    recording_mimi = RecordingMimi()
    model.mimi = recording_mimi  # ty: ignore[invalid-assignment]
    model.flow_lm = SimpleNamespace(  # ty: ignore[invalid-assignment]
        emb_std=2.0, emb_mean=1.0
    )
    model.config = SimpleNamespace(  # ty: ignore[invalid-assignment]
        mimi=SimpleNamespace(sample_rate=4)
    )

    alignment = WordAlignment(_units("one"))
    latents_queue = queue.Queue()
    result_queue = queue.Queue()
    latent = torch.arange(3, dtype=torch.float32).view(1, 1, 3)
    latents_queue.put((latent, torch.ones(1)))
    latents_queue.put(None)

    with patch("pocket_tts_timestamped.models.tts_model.is_voiced", return_value=True):
        model._decode_timestamped_audio_worker(
            latents_queue,
            result_queue,
            mimi_sequence_length=2,
            mimi_steps_per_latent=1,
            alignment=alignment,
            time_offset=0.0,
            cancel_event=threading.Event(),
        )

    assert len(recording_mimi.decoded_latents) == 1
    torch.testing.assert_close(recording_mimi.decoded_latents[0], latent * 2.0 + 1.0)
    assert recording_mimi.decoded_latents[0].shape == (1, 1, 3)
    results = list(result_queue.queue)
    assert any(kind == "event" and isinstance(value, AudioChunk) for kind, value in results)
    assert results[-1] == ("done", 1.0)


def test_selected_attention_matches_manual_text_softmax_and_preserves_output():
    torch.manual_seed(0)
    attention = StreamingMultiheadAttention(
        embed_dim=8, num_heads=2, rope=RotaryEmbedding(max_period=10_000)
    )
    attention._module_absolute_name = "attention"
    state = {"attention": attention.init_state(batch_size=1, sequence_length=8)}
    prompt = torch.randn(1, 2, 8)
    with torch.no_grad():
        attention(prompt, state)
    attention.increment_step(state["attention"], increment=2)

    query = torch.randn(1, 1, 8)
    captured_state = copy.deepcopy(state)
    baseline_state = copy.deepcopy(state)
    capture = SelectedAttentionCapture([(0, 1)], 0, 2, torch.eye(2))
    capture.begin_frame()
    captured_output = attention(query, captured_state, attention_capture=capture, layer_index=0)
    scores = capture.finish_frame()
    baseline_output = attention(query, baseline_state)

    projected = attention.in_proj(query).view(1, 1, 3, 2, 4)
    q, _, _ = torch.unbind(projected, dim=2)
    cached_k = state["attention"]["cache"][0, :, :2].permute(0, 2, 1, 3)
    q, _ = attention.rope(q, q, offset=state["attention"]["offset"].view(-1)[0])
    logits = torch.einsum("bd,btd->bt", q[:, 0, 1], cached_k[:, 1])
    expected = torch.softmax(logits / 2.0, dim=-1)[0]

    torch.testing.assert_close(scores, expected)
    torch.testing.assert_close(captured_output, baseline_output)


def test_selected_attention_supports_multiple_heads_in_one_layer():
    torch.manual_seed(1)
    attention = StreamingMultiheadAttention(
        embed_dim=8, num_heads=2, rope=RotaryEmbedding(max_period=10_000)
    )
    attention._module_absolute_name = "attention"
    state = {"attention": attention.init_state(batch_size=1, sequence_length=8)}
    prompt = torch.randn(1, 2, 8)
    with torch.no_grad():
        attention(prompt, state)
    attention.increment_step(state["attention"], increment=2)

    query = torch.randn(1, 1, 8)
    captured_state = copy.deepcopy(state)
    capture = SelectedAttentionCapture([(0, 0), (0, 1)], 0, 2, torch.eye(2))
    capture.begin_frame()
    attention(query, captured_state, attention_capture=capture, layer_index=0)
    scores = capture.finish_frame()

    projected = attention.in_proj(query).view(1, 1, 3, 2, 4)
    q, _, _ = torch.unbind(projected, dim=2)
    cached_k = state["attention"]["cache"][0, :, :2].permute(0, 2, 1, 3)
    q, _ = attention.rope(q, q, offset=state["attention"]["offset"].view(-1)[0])
    logits = torch.einsum("bhd,bhtd->bht", q[:, 0], cached_k)
    expected = torch.softmax(logits / 2.0, dim=-1).mean(dim=1)[0]
    torch.testing.assert_close(scores, expected)


def test_token_aggregation_normalizes_each_head_then_averages_equally():
    capture = SelectedAttentionCapture(
        [(0, 0), (1, 1)], 0, 3, torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    )
    capture.begin_frame()
    capture.record(0, (0,), torch.tensor([[[0.25, 0.25, 0.5]]]))
    capture.record(1, (1,), torch.tensor([[[0.1, 0.2, 0.7]]]))
    torch.testing.assert_close(capture.finish_frame(), torch.tensor([0.4, 0.6]))


def test_selected_attention_supports_dynamic_int8_projections():
    attention = StreamingMultiheadAttention(
        embed_dim=8, num_heads=2, rope=RotaryEmbedding(max_period=10_000)
    ).eval()
    with pytest.warns(DeprecationWarning):
        attention = torch.ao.quantization.quantize_dynamic(  # ty: ignore[deprecated]
            attention, {torch.nn.Linear}, dtype=torch.qint8
        )
    attention._module_absolute_name = "attention"
    state = {"attention": attention.init_state(batch_size=1, sequence_length=4)}
    capture = SelectedAttentionCapture([(0, 0)], 0, 1, torch.ones(1, 1))
    with torch.no_grad():
        attention(torch.randn(1, 1, 8), state)
        attention.increment_step(state["attention"])
        capture.begin_frame()
        output = attention(torch.randn(1, 1, 8), state, attention_capture=capture, layer_index=0)
    assert output.shape == (1, 1, 8)
    torch.testing.assert_close(capture.finish_frame(), torch.ones(1))


def test_timestamped_non_streaming_audio_concatenates_audio_events():
    model = object.__new__(TTSModel)
    torch.nn.Module.__init__(model)

    def fake_stream(**_kwargs):
        def events():
            yield WordStart("one", 0, 0.0)
            yield AudioChunk(torch.tensor([1.0, 2.0]), 0.0, 0.08)
            yield WordEnd("one", 0, 0.0, 0.16)
            yield AudioChunk(torch.tensor([3.0]), 0.08, 0.16)

        return events()

    model.generate_audio_with_timestamps_stream = fake_stream  # ty: ignore[invalid-assignment]
    result = TTSModel.generate_audio_with_timestamps(model, {}, "one")
    assert isinstance(result, TimestampedAudio)
    torch.testing.assert_close(result.audio, torch.tensor([1.0, 2.0, 3.0]))
    assert result.words == (WordTimestamp("one", 0, 0.0, 0.16),)


def test_unsupported_config_fails_before_generation():
    model = object.__new__(TTSModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(timestamp_heads=None)  # ty: ignore[invalid-assignment]
    with pytest.raises(ValueError, match="timestamp_heads is absent"):
        model.generate_audio_with_timestamps_stream({}, "one")


@pytest.mark.parametrize("append_terminal_punctuation", [True, False])
def test_timestamp_generation_forwards_terminal_punctuation_setting(append_terminal_punctuation):
    model = object.__new__(TTSModel)
    torch.nn.Module.__init__(model)
    model.model_recommended_frames_after_eos = None
    model.pad_with_spaces_for_short_inputs = False
    model.remove_semicolons = False
    model.append_terminal_punctuation = append_terminal_punctuation
    model.flow_lm = SimpleNamespace(  # ty: ignore[invalid-assignment]
        conditioner=SimpleNamespace(tokenizer=SimpleNamespace(sp=object()))
    )
    chunk = TimestampTextChunk("One", _units("One"), torch.empty(0, 1))

    def short_text(**kwargs):
        time_offset = kwargs["time_offset"]
        yield AudioChunk(torch.ones(1), time_offset, time_offset + 0.08)
        return time_offset + 0.08

    model._generate_audio_with_timestamps_short_text = short_text  # ty: ignore[invalid-assignment]
    with (
        patch(
            "pocket_tts_timestamped.models.tts_model.split_into_best_sentences",
            return_value=[chunk.text],
        ) as split_text,
        patch(
            "pocket_tts_timestamped.models.tts_model.prepare_text_prompt",
            return_value=(chunk.text, 1),
        ) as prepare_text,
        patch(
            "pocket_tts_timestamped.models.tts_model._iter_timestamp_text_chunks",
            return_value=iter([chunk]),
        ),
    ):
        list(TTSModel._generate_audio_with_timestamps_events(model, {}, "one", 50, None, True))

    split_text.assert_called_once_with(
        model.flow_lm.conditioner.tokenizer,
        "one",
        50,
        False,
        remove_semicolons=False,
        append_terminal_punctuation=append_terminal_punctuation,
    )
    prepare_text.assert_called_once_with("One", False, False, append_terminal_punctuation)


def test_chunk_event_offsets_and_global_word_indices():
    model = object.__new__(TTSModel)
    torch.nn.Module.__init__(model)
    model.model_recommended_frames_after_eos = None
    model.pad_with_spaces_for_short_inputs = False
    model.remove_semicolons = False
    model.append_terminal_punctuation = True
    model.flow_lm = SimpleNamespace(  # ty: ignore[invalid-assignment]
        conditioner=SimpleNamespace(tokenizer=SimpleNamespace(sp=object()))
    )
    chunks = [
        TimestampTextChunk("One.", _units("One", "P:."), torch.empty(0, 2)),
        TimestampTextChunk(
            "Two.",
            (_TextUnit("Two", 0, 3, _SourceWord("Two", 1, 5, 8)), _TextUnit(".", 3, 4, None)),
            torch.empty(0, 2),
        ),
    ]

    def short_text(**kwargs):
        timestamp_chunk = kwargs["timestamp_chunk"]
        time_offset = kwargs["time_offset"]
        word = timestamp_chunk.words[0]
        yield WordStart(word.text, word.word_index, time_offset)
        yield AudioChunk(torch.ones(2), time_offset, time_offset + 0.08)
        return time_offset + 0.08

    model._generate_audio_with_timestamps_short_text = short_text  # ty: ignore[invalid-assignment]
    with (
        patch(
            "pocket_tts_timestamped.models.tts_model.split_into_best_sentences",
            return_value=["One.", "Two."],
        ),
        patch(
            "pocket_tts_timestamped.models.tts_model._iter_timestamp_text_chunks",
            return_value=iter(chunks),
        ),
    ):
        generator = TTSModel._generate_audio_with_timestamps_events(
            model, {}, "One. Two.", max_tokens=50, frames_after_eos=None, copy_state=True
        )
        events = list(generator)

    assert [event.word_index for event in events if isinstance(event, WordStart)] == [0, 1]
    audio_events = [event for event in events if isinstance(event, AudioChunk)]
    assert [(event.start_time, event.end_time) for event in audio_events] == [
        (0.0, 0.08),
        (0.08, 0.16),
    ]
    assert audio_events[-1].end_time == 0.16


def test_degraded_word_gaps_preserve_audio_and_event_order():
    model = object.__new__(TTSModel)
    torch.nn.Module.__init__(model)
    model.model_recommended_frames_after_eos = None
    model.pad_with_spaces_for_short_inputs = False
    model.remove_semicolons = False
    model.append_terminal_punctuation = True
    model.flow_lm = SimpleNamespace(  # ty: ignore[invalid-assignment]
        conditioner=SimpleNamespace(tokenizer=SimpleNamespace(sp=object()))
    )
    chunk = TimestampTextChunk(
        "One different three.",
        (
            _TextUnit("One", 0, 3, _SourceWord("one", 0, 0, 3)),
            _TextUnit("different", 4, 13, None, synthetic=True),
            _TextUnit("three", 14, 19, _SourceWord("three", 2, 12, 17)),
            _TextUnit(".", 19, 20, None),
        ),
        torch.ones(1, 4),
    )

    def short_text(**kwargs):
        timestamp_chunk = kwargs["timestamp_chunk"]
        time_offset = kwargs["time_offset"]
        for word in timestamp_chunk.words:
            yield WordStart(word.text, word.word_index, time_offset)
        yield AudioChunk(torch.tensor([1.0, 2.0]), time_offset, time_offset + 0.08)
        return time_offset + 0.08

    model._generate_audio_with_timestamps_short_text = short_text  # ty: ignore[invalid-assignment]
    with (
        patch(
            "pocket_tts_timestamped.models.tts_model.split_into_best_sentences",
            return_value=[chunk.text],
        ),
        patch(
            "pocket_tts_timestamped.models.tts_model._iter_timestamp_text_chunks",
            return_value=iter([chunk]),
        ),
    ):
        events = list(
            TTSModel._generate_audio_with_timestamps_events(
                model, {}, "one missing three", 50, None, True
            )
        )

    word_events = [event for event in events if isinstance(event, WordStart)]
    audio_index = next(index for index, event in enumerate(events) if isinstance(event, AudioChunk))
    assert [event.word_index for event in word_events] == [0, 2]
    assert all(events.index(event) < audio_index for event in word_events)
    audio_event = events[audio_index]
    assert isinstance(audio_event, AudioChunk)
    torch.testing.assert_close(audio_event.audio, torch.tensor([1.0, 2.0]))


def test_timestamp_head_config_validation():
    config = load_config(CONFIGS_DIR / "english.yaml").model_dump()
    transformer = config["flow_lm"]["transformer"]
    last_layer = transformer["num_layers"] - 1
    last_head = transformer["num_heads"] - 1

    config["timestamp_heads"] = [{"layer": last_layer, "head": last_head}]
    validated = Config(**config)
    assert validated.timestamp_heads is not None
    assert validated.timestamp_heads[0].layer == last_layer
    assert validated.timestamp_heads[0].head == last_head

    config["timestamp_heads"] = [{"layer": transformer["num_layers"], "head": 0}]
    with pytest.raises(ValueError, match="outside the FlowLM layer range"):
        Config(**config)

    config["timestamp_heads"] = [{"layer": 0, "head": transformer["num_heads"]}]
    with pytest.raises(ValueError, match="outside the FlowLM head range"):
        Config(**config)

    config["timestamp_heads"] = [{"layer": 0, "head": 0}, {"layer": 0, "head": 0}]
    with pytest.raises(ValueError, match="Duplicate timestamp head"):
        Config(**config)


@pytest.fixture(scope="module")
def spanish_timestamp_tokenizer():
    config = load_config(CONFIGS_DIR / "spanish.yaml")
    tokenizer_path = download_if_necessary(config.flow_lm.lookup_table.tokenizer_path)
    return sentencepiece.SentencePieceProcessor(str(tokenizer_path))


@pytest.fixture(scope="module")
def spanish_timestamp_model():
    return TTSModel.load_model(language="spanish")


@pytest.fixture(scope="module")
def spanish_timestamp_model_and_voice_state(spanish_timestamp_model):
    model = spanish_timestamp_model
    sample_positions = torch.arange(model.sample_rate, dtype=torch.float32)
    audio_prompt = (0.1 * torch.sin(2 * torch.pi * 220 * sample_positions / model.sample_rate))[
        None
    ]
    voice_state = model.get_state_for_audio_prompt(audio_prompt)
    return model, voice_state


@pytest.mark.parametrize(
    ("source_text", "expected_text", "expected_words"),
    [
        ("A ﬁne result.", "A fine result.", ("A", "ﬁne", "result")),
        (unicodedata.normalize("NFD", "Café naïve."), "Café naïve.", ("Cafe\u0301", "nai\u0308ve")),
    ],
)
def test_timestamp_chunks_use_production_tokenizer_for_canonical_unicode(
    spanish_timestamp_tokenizer, source_text, expected_text, expected_words
):
    chunk = build_timestamp_text_chunks(source_text, [source_text], spanish_timestamp_tokenizer)[0]
    expected_token_ids = spanish_timestamp_tokenizer.encode(chunk.text, out_type=int)

    assert chunk.text == expected_text
    assert chunk.token_to_unit.shape[0] == len(expected_token_ids)
    assert chunk.prepared_tokens is not None
    assert chunk.prepared_tokens[0].tolist() == expected_token_ids
    assert [word.text for word in chunk.words] == list(expected_words)


def test_timestamp_generation_end_to_end_with_accented_text(
    spanish_timestamp_model_and_voice_state,
):
    model, voice_state = spanish_timestamp_model_and_voice_state
    stream = model.generate_audio_with_timestamps_stream(
        voice_state, "El café está aquí.", frames_after_eos=0
    )
    events = list(stream)

    word_events = [event for event in events if isinstance(event, WordEnd)]
    expected_words = ["El", "café", "está", "aquí"]
    observed_words = [(event.word_index, event.word) for event in word_events]
    assert observed_words
    assert observed_words == list(enumerate(expected_words))[: len(observed_words)]
    audio_events = [event for event in events if isinstance(event, AudioChunk)]
    assert audio_events
    assert all(event.end_time > event.start_time for event in audio_events)
