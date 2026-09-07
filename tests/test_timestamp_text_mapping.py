# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, ANN204
import re
import unicodedata
from unittest.mock import patch

import pytest
import torch

from pocket_tts_timestamped.models.tts_model import split_into_best_sentences
from pocket_tts_timestamped.modules.text_conditioner import LUTConditioner
from pocket_tts_timestamped.timestamps import build_timestamp_text_chunks
from pocket_tts_timestamped.timestamps.text import _iter_timestamp_text_chunks
from pocket_tts_timestamped.utils.config import CONFIGS_DIR, load_config

BYTE_PIECE = re.compile(r"<0x[0-9A-F]{2}>")


@pytest.fixture(scope="module")
def conditioner():
    lookup_table = load_config(CONFIGS_DIR / "english.yaml").flow_lm.lookup_table
    return LUTConditioner(lookup_table.n_bins, lookup_table.tokenizer_path, dim=1, output_dim=1)


def _timestamp_chunks(conditioner, source_text, *, max_tokens=100, remove_semicolons=False):
    chunks = split_into_best_sentences(
        conditioner.tokenizer,
        source_text,
        max_tokens,
        pad_with_spaces_for_short_inputs=False,
        remove_semicolons=remove_semicolons,
    )
    return build_timestamp_text_chunks(source_text, chunks, conditioner.tokenizer.sp)


def _assert_mapping_contract(
    conditioner, source_text, expected_words, *, prepared_chunks=None, **kwargs
):
    chunks = (
        _timestamp_chunks(conditioner, source_text, **kwargs)
        if prepared_chunks is None
        else build_timestamp_text_chunks(source_text, prepared_chunks, conditioner.tokenizer.sp)
    )
    words = [word for chunk in chunks for word in chunk.words]

    assert [word.text for word in words] == list(expected_words)
    assert [word.word_index for word in words] == list(range(len(expected_words)))

    for word in words:
        assert source_text[word.begin : word.end] == word.text

    for chunk in chunks:
        prepared = conditioner.prepare(chunk.text)
        expected_ids = conditioner.tokenizer.sp.encode(chunk.text, out_type=int)
        assert prepared[0].tolist() == expected_ids
        assert chunk.prepared_tokens is not None
        assert chunk.prepared_tokens[0].tolist() == expected_ids
        assert chunk.token_to_unit.shape == (len(expected_ids), len(chunk.units))
        assert torch.isfinite(chunk.token_to_unit).all()
        assert (chunk.token_to_unit >= 0).all()

        row_sums = chunk.token_to_unit.sum(dim=1)
        assert torch.all(
            torch.isclose(row_sums, torch.zeros_like(row_sums))
            | torch.isclose(row_sums, torch.ones_like(row_sums))
        )

        previous_end = 0
        for unit in chunk.units:
            assert 0 <= unit.chunk_begin < unit.chunk_end <= len(chunk.text)
            assert unit.chunk_begin >= previous_end
            previous_end = unit.chunk_end

        for unit_index, _unit in enumerate(chunk.units):
            assert chunk.token_to_unit[:, unit_index].sum() > 0

    return chunks


@pytest.mark.parametrize(
    ("source_text", "expected_words", "remove_semicolons"),
    [
        ("  hello\n\r  world  ", ("hello", "world"), False),
        ("first; second", ("first", "second"), True),
        ("Café naïve.", ("Café", "naïve"), False),
        (unicodedata.normalize("NFD", "Café naïve."), ("Cafe\u0301", "nai\u0308ve"), False),
        ("A ﬁne result.", ("A", "ﬁne", "result"), False),
        ("Ｆｕｌｌ ① test.", ("Ｆｕｌｌ", "①", "test"), False),
        ("Roman Ⅷ test.", ("Roman", "Ⅷ", "test"), False),
        ("5㈠ test.", ("5㈠", "test"), False),
        ("It is 20℃.", ("It", "is", "20"), False),
        ("Room №5.", ("Room", "5"), False),
        ("Meet at ㏂ 10.", ("Meet", "at", "10"), False),
        ("ıstanbul is here.", ("ıstanbul", "is", "here"), False),
        ("ßtraße ǆungla σς.", ("ßtraße", "ǆungla", "σς"), False),
        ("a\u0327\u0301 la carte.", ("a\u0327\u0301", "la", "carte"), False),
        ("l’esprit isn't blue‑green.", ("l’esprit", "isn't", "blue‑green"), False),
        ("Price_1 is 3.14—okay...", ("Price", "1", "is", "3", "14", "okay"), False),
        ("Pay €5 😀 now。", ("Pay", "5", "now"), False),
        ("zero\u200bwidth text.", ("zero", "width", "text"), False),
        ("😀 €", (), False),
    ],
    ids=[
        "preparation-whitespace",
        "semicolon-replacement",
        "nfc",
        "nfd",
        "ligature",
        "full-width-and-circled",
        "roman-numeral",
        "parenthesized-ideograph",
        "degree-celsius",
        "numero-sign",
        "am-symbol",
        "initial-dotless-i",
        "case-expansions",
        "multiple-combining-marks",
        "apostrophes-and-hyphens",
        "word-boundaries",
        "symbols",
        "zero-width-boundary",
        "no-lexical-words",
    ],
)
def test_curated_text_mapping_contract(conditioner, source_text, expected_words, remove_semicolons):
    _assert_mapping_contract(
        conditioner, source_text, expected_words, remove_semicolons=remove_semicolons
    )


@pytest.mark.parametrize("source_text", ["A中B", "A𐐀B", "a\u0301\u0300"])
def test_every_byte_fallback_piece_inside_a_word_is_mapped(conditioner, source_text):
    chunks = _assert_mapping_contract(
        conditioner, source_text, tuple(word for word in source_text.split() if word)
    )
    assert len(chunks) == 1
    chunk = chunks[0]
    pieces = conditioner.tokenizer.sp.encode(chunk.text, out_type=str)
    byte_piece_indices = [
        index for index, piece in enumerate(pieces) if BYTE_PIECE.fullmatch(piece)
    ]

    assert byte_piece_indices
    word_unit = next(index for index, unit in enumerate(chunk.units) if unit.is_word)
    assert torch.equal(
        chunk.token_to_unit[byte_piece_indices, word_unit], torch.ones(len(byte_piece_indices))
    )


def test_canonical_equivalents_have_the_same_normalized_mapping(conditioner):
    nfc_chunks = _assert_mapping_contract(conditioner, "Café naïve.", ("Café", "naïve"))
    nfd_chunks = _assert_mapping_contract(
        conditioner, unicodedata.normalize("NFD", "Café naïve."), ("Cafe\u0301", "nai\u0308ve")
    )

    assert [chunk.text for chunk in nfc_chunks] == [chunk.text for chunk in nfd_chunks]
    for nfc, nfd in zip(nfc_chunks, nfd_chunks):
        torch.testing.assert_close(nfc.token_to_unit, nfd.token_to_unit)


def test_compatibility_punctuation_from_inside_a_word_stays_in_that_unit(conditioner):
    chunk = _assert_mapping_contract(conditioner, "5㈠ test.", ("5㈠", "test"))[0]
    first_word_unit = next(unit for unit in chunk.units if unit.is_word)
    first_word = first_word_unit.word

    assert first_word is not None
    assert "5㈠ test."[first_word.begin : first_word.end] == "5㈠"
    assert chunk.text[first_word_unit.chunk_begin : first_word_unit.chunk_end] == "5(一)"
    assert [unit.text for unit in chunk.units if not unit.is_word] == ["."]


@pytest.mark.parametrize(
    ("source_text", "expected_punctuation", "expected_synthetic"),
    [("hello world", ".", True), ("hello world!", "!", False)],
)
def test_only_preparation_added_terminal_punctuation_is_synthetic(
    conditioner, source_text, expected_punctuation, expected_synthetic
):
    chunk = _timestamp_chunks(conditioner, source_text)[0]
    punctuation = [unit for unit in chunk.units if not unit.is_word]

    assert [(unit.text, unit.synthetic) for unit in punctuation] == [
        (expected_punctuation, expected_synthetic)
    ]


def test_compatibility_punctuation_can_split_a_source_word_between_chunks(conditioner):
    chunks = _assert_mapping_contract(
        conditioner, "A⒚B 10.", ("A⒚", "B", "10"), prepared_chunks=["A19.", "B 10."]
    )

    assert [chunk.text for chunk in chunks] == ["A19.", "B 10."]


def test_compatibility_character_that_normalizes_to_only_a_mark_is_not_a_word(conditioner):
    _assert_mapping_contract(conditioner, "X ﾞ Y.", ("X", "Y"), max_tokens=8)


def test_chunking_does_not_change_public_word_sequence(conditioner):
    source_text = "Room №5. Meet at ㏂ 10. Café naïve; blue-green world!"
    expected = ("Room", "5", "Meet", "at", "10", "Café", "naïve", "blue-green", "world")

    results = []
    for max_tokens in (8, 20, 100):
        chunks = _assert_mapping_contract(conditioner, source_text, expected, max_tokens=max_tokens)
        results.append([(word.word_index, word.text) for chunk in chunks for word in chunk.words])

    assert results[0] == results[1] == results[2]


@pytest.mark.parametrize("source_text", ["", " ", "\n\r"])
def test_empty_text_has_the_same_rejection_as_ordinary_preprocessing(conditioner, source_text):
    with pytest.raises(ValueError, match="empty"):
        _timestamp_chunks(conditioner, source_text)


def test_mapping_rejects_prepared_text_from_a_different_source(conditioner):
    with pytest.raises(ValueError, match="does not match"):
        build_timestamp_text_chunks("source words", ["Different words."], conditioner.tokenizer.sp)


@pytest.mark.parametrize(
    "source_text", ["Café naïve.", unicodedata.normalize("NFD", "Café naïve."), "A ﬁne result."]
)
def test_common_unicode_mapping_does_not_enter_robust_fallback(conditioner, source_text):
    chunks = split_into_best_sentences(
        conditioner.tokenizer,
        source_text,
        100,
        pad_with_spaces_for_short_inputs=False,
        remove_semicolons=False,
    )
    with patch(
        "pocket_tts_timestamped.timestamps.text._map_source_words_to_chunks",
        side_effect=AssertionError("robust fallback should not run"),
    ):
        assert build_timestamp_text_chunks(source_text, chunks, conditioner.tokenizer.sp)


def test_timestamp_chunk_mapping_tokenizes_later_chunks_lazily(conditioner):
    class CountingProcessor:
        def __init__(self, processor):
            self.processor = processor
            self.encoded_texts = []

        def encode(self, text, *args, **kwargs):
            self.encoded_texts.append(text)
            return self.processor.encode(text, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.processor, name)

    processor = CountingProcessor(conditioner.tokenizer.sp)
    iterator = _iter_timestamp_text_chunks("one two", ["One", "two."], processor, best_effort=True)
    assert processor.encoded_texts == []

    first = next(iterator)
    assert first.text == "One"
    assert processor.encoded_texts
    assert set(processor.encoded_texts) == {"One"}

    second = next(iterator)
    assert second.text == "two."
    assert "two." in processor.encoded_texts


def test_product_mapping_degrades_to_unambiguous_gaps(conditioner):
    with (
        patch(
            "pocket_tts_timestamped.timestamps.text._map_source_words_to_chunks",
            side_effect=ValueError("forced robust failure"),
        ),
        patch("pocket_tts_timestamped.timestamps.text.logger.warning") as warning,
    ):
        chunks = list(
            _iter_timestamp_text_chunks(
                "one missing three",
                ["One different three."],
                conditioner.tokenizer.sp,
                best_effort=True,
            )
        )

    assert [word.word_index for word in chunks[0].words] == [0, 2]
    ignored = [unit for unit in chunks[0].units if unit.synthetic and not unit.is_word]
    assert any(unit.text == "different" for unit in ignored)
    warning.assert_called_once_with(
        "Timestamp text mapping omitted %d source word(s); audio generation continues", 1
    )
