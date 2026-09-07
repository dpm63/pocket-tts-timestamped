from __future__ import annotations

import logging
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import groupby
from typing import Literal, Protocol, cast

import torch


@dataclass(frozen=True, slots=True)
class _SourceWord:
    text: str
    word_index: int
    begin: int
    end: int


@dataclass(frozen=True, slots=True)
class _TextUnit:
    text: str
    chunk_begin: int
    chunk_end: int
    word: _SourceWord | None
    synthetic: bool = False

    @property
    def is_word(self) -> bool:
        return self.word is not None


@dataclass(frozen=True, slots=True)
class TimestampTextChunk:
    text: str
    units: tuple[_TextUnit, ...]
    token_to_unit: torch.Tensor
    prepared_tokens: torch.Tensor | None = None

    @property
    def words(self) -> tuple[_SourceWord, ...]:
        return tuple(unit.word for unit in self.units if unit.word is not None)


@dataclass(frozen=True, slots=True)
class _Span:
    begin: int
    end: int


@dataclass(frozen=True, slots=True)
class _CanonicalProjection:
    text: str
    origins: tuple[_Span, ...]


@dataclass(frozen=True, slots=True)
class _ChunkOrigin:
    chunk_index: int
    chunk_begin: int
    chunk_end: int


@dataclass(frozen=True, slots=True)
class _CanonicalGroup:
    chunk_index: int
    canonical_begin: int
    canonical_end: int


@dataclass(frozen=True, slots=True)
class _WordPart:
    text: str
    source_begin: int
    source_end: int
    group: _CanonicalGroup


@dataclass(frozen=True, slots=True)
class _MappedWord:
    word: _SourceWord
    chunk_span: _Span


@dataclass(frozen=True, slots=True)
class _PreparedWord:
    chunk_index: int
    span: _Span
    normalized: str


@dataclass(frozen=True, slots=True)
class _PieceLayout:
    spans: tuple[_Span, ...]
    coordinate_system: Literal["byte", "codepoint"]
    token_ids: tuple[int, ...]


_FOLD_TRANSLATION = str.maketrans({"’": "'", "‐": "-", "‑": "-"})
_CANONICAL_WORD_CHARACTERS = frozenset("'-")
_WORD_CONTINUATIONS = frozenset("-‐‑'’")

logger = logging.getLogger(__name__)


class _ImmutablePiece(Protocol):
    begin: int
    end: int
    id: int
    piece: str


class _ImmutableProto(Protocol):
    pieces: Iterable[_ImmutablePiece]


class _SentencePieceProcessor(Protocol):
    def encode(self, text: str, *args: object, **kwargs: object) -> object: ...

    def id_to_piece(self, token_id: object) -> str: ...


def _fold(text: str) -> str:
    return text.casefold().translate(_FOLD_TRANSLATION)


def _is_canonical_word_character(character: str) -> bool:
    return character.isalnum() or character in _CANONICAL_WORD_CHARACTERS


def _normalized_word(text: str) -> str:
    normalized = _fold(unicodedata.normalize("NFKC", text))
    return "".join(character for character in normalized if _is_canonical_word_character(character))


def _is_word_character(character: str) -> bool:
    return character.isalnum()


def _is_combining_mark(character: str) -> bool:
    return unicodedata.category(character).startswith("M")


def _lexical_word_spans(text: str) -> list[_Span]:
    """Return lexical spans, retaining combining marks with their base character."""

    spans: list[_Span] = []
    index = 0
    while index < len(text):
        if not _is_word_character(text[index]):
            index += 1
            continue

        begin = index
        while True:
            while index < len(text) and (
                _is_word_character(text[index]) or _is_combining_mark(text[index])
            ):
                index += 1
            if (
                index + 1 < len(text)
                and text[index] in _WORD_CONTINUATIONS
                and _is_word_character(text[index + 1])
            ):
                index += 1
                continue
            break
        if _normalized_word(text[begin:index]):
            spans.append(_Span(begin, index))
    return spans


def _lexical_words(text: str) -> list[_SourceWord]:
    return [
        _SourceWord(
            text=text[span.begin : span.end], word_index=index, begin=span.begin, end=span.end
        )
        for index, span in enumerate(_lexical_word_spans(text))
    ]


def _byte_piece_value(piece: str) -> int | None:
    if len(piece) != 6 or not piece.startswith("<0x") or not piece.endswith(">"):
        return None
    try:
        return int(piece[3:5], 16)
    except ValueError:
        return None


def _utf8_sequence_length(first_byte: int) -> int:
    if first_byte < 0x80:
        return 1
    if 0xC2 <= first_byte <= 0xDF:
        return 2
    if 0xE0 <= first_byte <= 0xEF:
        return 3
    if 0xF0 <= first_byte <= 0xF4:
        return 4
    return 1


def _expand_byte_fallback_spans(
    spans: tuple[_Span, ...], pieces: tuple[str, ...]
) -> tuple[_Span, ...]:
    """Give every byte piece for one Unicode scalar the scalar's complete span."""

    expanded = list(spans)
    index = 0
    while index < len(pieces):
        first_byte = _byte_piece_value(pieces[index])
        if first_byte is None:
            index += 1
            continue

        sequence_length = _utf8_sequence_length(first_byte)
        byte_values = [
            _byte_piece_value(piece) for piece in pieces[index : index + sequence_length]
        ]
        continuation_bytes = byte_values[1:]
        valid_sequence = (
            len(byte_values) == sequence_length
            and all(value is not None for value in byte_values)
            and all(value is not None and 0x80 <= value <= 0xBF for value in continuation_bytes)
        )
        end = index + sequence_length if valid_sequence else index + 1

        begin_offset = min(span.begin for span in spans[index:end])
        end_offset = max(span.end for span in spans[index:end])
        if end_offset > begin_offset:
            expanded[index:end] = [_Span(begin_offset, end_offset)] * (end - index)
        index = end
    return tuple(expanded)


class _SentencePieceLayoutReader:
    def __init__(self, processor: object):
        self._processor = cast(_SentencePieceProcessor, processor)
        self._reader: Callable[[str], _PieceLayout] | None = None

    def read(self, text: str) -> _PieceLayout:
        if self._reader is not None:
            return self._reader(text)

        try:
            encoded = self._processor.encode(text, return_type="offset_mapping", return_bytes=True)
        except TypeError:
            self._reader = self._from_immutable_proto
            return self._reader(text)

        self._reader = self._from_offset_mapping
        return self._parse_offset_mapping(encoded)

    def _from_offset_mapping(self, text: str) -> _PieceLayout:
        encoded = self._processor.encode(text, return_type="offset_mapping", return_bytes=True)
        return self._parse_offset_mapping(encoded)

    def _parse_offset_mapping(self, encoded: object) -> _PieceLayout:
        if not isinstance(encoded, dict):
            raise TypeError("SentencePiece offset_mapping must return a dictionary")
        offsets = encoded.get("offsets")
        token_ids = encoded.get("ids")
        if not isinstance(offsets, list) or not isinstance(token_ids, list):
            raise TypeError("SentencePiece offset_mapping must contain list ids and offsets")
        if len(offsets) != len(token_ids):
            raise ValueError("SentencePiece offset_mapping ids and offsets have different lengths")
        try:
            spans = tuple(_Span(int(begin), int(end)) for begin, end in offsets)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "SentencePiece offset_mapping offsets must be begin/end pairs"
            ) from error

        pieces = encoded.get("pieces")
        if not isinstance(pieces, list):
            pieces = [self._processor.id_to_piece(token_id) for token_id in token_ids]
        if len(pieces) != len(spans):
            raise ValueError(
                "SentencePiece offset_mapping pieces and offsets have different lengths"
            )
        piece_texts = tuple(
            piece.decode("utf-8") if isinstance(piece, bytes) else str(piece) for piece in pieces
        )
        return _PieceLayout(
            _expand_byte_fallback_spans(spans, piece_texts),
            "byte",
            tuple(int(token_id) for token_id in token_ids),
        )

    def _from_immutable_proto(self, text: str) -> _PieceLayout:
        proto = cast(_ImmutableProto, self._processor.encode(text, out_type="immutable_proto"))
        spans = tuple(_Span(int(piece.begin), int(piece.end)) for piece in proto.pieces)
        pieces = tuple(str(piece.piece) for piece in proto.pieces)
        token_ids = tuple(int(piece.id) for piece in proto.pieces)
        return _PieceLayout(_expand_byte_fallback_spans(spans, pieces), "codepoint", token_ids)


def _canonical_characters(text: str) -> _CanonicalProjection:
    """Return canonical text and each character's span in the input text."""

    normalized = ""
    normalized_origins: list[_Span] = []

    for index in range(len(text)):
        updated = unicodedata.normalize("NFKC", text[: index + 1])
        common = 0
        while (
            common < len(normalized)
            and common < len(updated)
            and normalized[common] == updated[common]
        ):
            common += 1
        changed_origins = [*normalized_origins[common:], _Span(index, index + 1)]
        changed_origin = _Span(
            min(origin.begin for origin in changed_origins),
            max(origin.end for origin in changed_origins),
        )
        normalized_origins = [
            *normalized_origins[:common],
            *([changed_origin] * (len(updated) - common)),
        ]
        normalized = updated

    characters: list[str] = []
    origins: list[_Span] = []
    for character, origin in zip(normalized, normalized_origins):
        for folded_character in _fold(character):
            if _is_canonical_word_character(folded_character):
                characters.append(folded_character)
                origins.append(origin)
    return _CanonicalProjection("".join(characters), tuple(origins))


def _capitalization_variants(text: str) -> tuple[str, ...]:
    if not text:
        return (text,)
    return tuple(dict.fromkeys((text, text[0].upper() + text[1:])))


def _surface_span(
    source_text: str, chunk_text: str, canonical_span: _Span, capitalize: bool
) -> _Span:
    candidates = _capitalization_variants(source_text) if capitalize else (source_text,)
    surface_matches: list[_Span] = []
    for candidate in candidates:
        surface = unicodedata.normalize("NFKC", candidate)
        surface_begin = chunk_text.find(surface)
        while surface_begin >= 0:
            surface_span = _Span(surface_begin, surface_begin + len(surface))
            if (
                surface_span.begin <= canonical_span.begin
                and canonical_span.end <= surface_span.end
            ):
                surface_matches.append(surface_span)
            surface_begin = chunk_text.find(surface, surface_begin + 1)

    if surface_matches:
        canonical_span = min(
            surface_matches,
            key=lambda span: (
                canonical_span.begin - span.begin + span.end - canonical_span.end,
                span.begin,
            ),
        )
    chunk_end = canonical_span.end
    while chunk_end < len(chunk_text) and _is_combining_mark(chunk_text[chunk_end]):
        chunk_end += 1
    return _Span(canonical_span.begin, chunk_end)


def _build_canonical_index(normalized_chunks: list[str]) -> tuple[str, tuple[_ChunkOrigin, ...]]:
    canonical_parts: list[str] = []
    origins: list[_ChunkOrigin] = []
    for chunk_index, chunk in enumerate(normalized_chunks):
        projection = _canonical_characters(chunk)
        canonical_parts.append(projection.text)
        origins.extend(
            _ChunkOrigin(chunk_index, origin.begin, origin.end) for origin in projection.origins
        )
    return "".join(canonical_parts), tuple(origins)


def _validate_canonical_source(source_text: str, canonical_text: str) -> None:
    stripped_source = source_text.strip()
    canonical_variants = {
        _normalized_word(variant) for variant in _capitalization_variants(stripped_source)
    }
    if canonical_text not in canonical_variants:
        raise ValueError("Prepared timestamp text does not match the original input")


def _find_word_in_canonical_text(
    canonical_text: str, candidates: tuple[str, ...], minimum_begin: int
) -> _Span:
    matches: list[_Span] = []
    for candidate in candidates:
        begin = canonical_text.find(candidate, minimum_begin)
        if begin >= 0:
            matches.append(_Span(begin, begin + len(candidate)))
    if not matches:
        raise ValueError("Prepared timestamp text does not contain an input word")
    return min(matches, key=lambda match: match.begin)


def _canonical_groups(
    canonical_span: _Span, origins: tuple[_ChunkOrigin, ...]
) -> tuple[_CanonicalGroup, ...]:
    groups = []
    positions = range(canonical_span.begin, canonical_span.end)
    for chunk_index, grouped_positions in groupby(
        positions, key=lambda position: origins[position].chunk_index
    ):
        grouped_positions = tuple(grouped_positions)
        groups.append(_CanonicalGroup(chunk_index, grouped_positions[0], grouped_positions[-1] + 1))
    return tuple(groups)


def _split_word_across_chunks(
    source_word: _SourceWord,
    canonical_span: _Span,
    source_projection: _CanonicalProjection,
    chunk_origins: tuple[_ChunkOrigin, ...],
) -> tuple[_WordPart, ...]:
    if len(source_projection.origins) != canonical_span.end - canonical_span.begin:
        raise ValueError(f"Cannot project input word {source_word.text!r} onto prepared text")

    groups = _canonical_groups(canonical_span, chunk_origins)
    source_begins = [
        min(
            origin.begin
            for origin in source_projection.origins[
                group.canonical_begin - canonical_span.begin : group.canonical_end
                - canonical_span.begin
            ]
        )
        for group in groups
    ]
    if any(
        next_begin <= current_begin
        for current_begin, next_begin in zip(source_begins, source_begins[1:])
    ):
        raise ValueError(f"Prepared chunks split input character in {source_word.text!r}")

    parts = []
    for group_index, group in enumerate(groups):
        relative_begin = 0 if group_index == 0 else source_begins[group_index]
        relative_end = (
            source_begins[group_index + 1]
            if group_index + 1 < len(groups)
            else len(source_word.text)
        )
        parts.append(
            _WordPart(
                text=source_word.text[relative_begin:relative_end],
                source_begin=source_word.begin + relative_begin,
                source_end=source_word.begin + relative_end,
                group=group,
            )
        )
    return tuple(parts)


def _map_source_words_to_chunks(
    source_text: str, normalized_chunks: list[str], source_words: list[_SourceWord]
) -> list[list[_MappedWord]]:
    canonical_text, chunk_origins = _build_canonical_index(normalized_chunks)
    _validate_canonical_source(source_text, canonical_text)

    mapped: list[list[_MappedWord]] = [[] for _ in normalized_chunks]
    minimum_begin = 0
    next_word_index = 0
    for source_word in source_words:
        canonical_word = _normalized_word(source_word.text)
        candidates = (canonical_word,)
        if source_word.word_index == 0:
            candidates = tuple(
                dict.fromkeys(
                    _normalized_word(variant)
                    for variant in _capitalization_variants(source_word.text)
                )
            )
        canonical_span = _find_word_in_canonical_text(canonical_text, candidates, minimum_begin)
        if canonical_span.begin == canonical_span.end:
            raise ValueError(f"Input word {source_word.text!r} has no timestamp text")

        source_projection = _canonical_characters(source_word.text)
        parts = _split_word_across_chunks(
            source_word, canonical_span, source_projection, chunk_origins
        )
        for part_index, part in enumerate(parts):
            word = _SourceWord(
                text=part.text,
                word_index=next_word_index,
                begin=part.source_begin,
                end=part.source_end,
            )
            next_word_index += 1
            canonical_chunk_span = _Span(
                chunk_origins[part.group.canonical_begin].chunk_begin,
                chunk_origins[part.group.canonical_end - 1].chunk_end,
            )
            chunk_span = _surface_span(
                part.text,
                normalized_chunks[part.group.chunk_index],
                canonical_chunk_span,
                capitalize=source_word.word_index == 0 and part_index == 0,
            )
            mapped[part.group.chunk_index].append(_MappedWord(word, chunk_span))
        minimum_begin = canonical_span.end
    return mapped


def _punctuation_spans(text: str, covered: list[bool]) -> list[_Span]:
    spans: list[_Span] = []
    index = 0
    while index < len(text):
        is_punctuation = not covered[index] and unicodedata.category(text[index]).startswith("P")
        if not is_punctuation:
            index += 1
            continue
        end = index + 1
        while (
            end < len(text) and not covered[end] and unicodedata.category(text[end]).startswith("P")
        ):
            end += 1
        spans.append(_Span(index, end))
        index = end
    return spans


def _is_synthetic_punctuation(
    source_text: str,
    source_words: list[_SourceWord],
    chunk_words: list[_MappedWord],
    punctuation_span: _Span,
) -> bool:
    if not chunk_words or punctuation_span.begin < chunk_words[-1].chunk_span.end:
        return False
    last_word = chunk_words[-1].word
    next_word_index = last_word.word_index + 1
    source_boundary_end = (
        source_words[next_word_index].begin
        if next_word_index < len(source_words)
        else len(source_text)
    )
    source_trailing_text = source_text[last_word.end : source_boundary_end]
    return not any(
        unicodedata.category(character).startswith("P") for character in source_trailing_text
    )


def _build_units(
    source_text: str,
    source_words: list[_SourceWord],
    normalized_chunk: str,
    chunk_words: list[_MappedWord],
    ignored_spans: tuple[_Span, ...] = (),
) -> tuple[_TextUnit, ...]:
    units: list[_TextUnit] = []
    covered = [False] * len(normalized_chunk)
    for mapped_word in chunk_words:
        span = mapped_word.chunk_span
        units.append(
            _TextUnit(
                text=mapped_word.word.text,
                chunk_begin=span.begin,
                chunk_end=span.end,
                word=mapped_word.word,
            )
        )
        covered[span.begin : span.end] = [True] * (span.end - span.begin)

    for span in ignored_spans:
        units.append(
            _TextUnit(
                text=normalized_chunk[span.begin : span.end],
                chunk_begin=span.begin,
                chunk_end=span.end,
                word=None,
                synthetic=True,
            )
        )
        covered[span.begin : span.end] = [True] * (span.end - span.begin)

    for span in _punctuation_spans(normalized_chunk, covered):
        units.append(
            _TextUnit(
                text=normalized_chunk[span.begin : span.end],
                chunk_begin=span.begin,
                chunk_end=span.end,
                word=None,
                synthetic=_is_synthetic_punctuation(source_text, source_words, chunk_words, span),
            )
        )
    units.sort(key=lambda unit: (unit.chunk_begin, unit.chunk_end))
    return tuple(units)


def _unit_spans(
    text: str, units: tuple[_TextUnit, ...], coordinate_system: Literal["byte", "codepoint"]
) -> list[_Span]:
    if coordinate_system == "codepoint":
        return [_Span(unit.chunk_begin, unit.chunk_end) for unit in units]

    byte_boundaries = [0]
    for character in text:
        byte_boundaries.append(byte_boundaries[-1] + len(character.encode("utf-8")))
    return [
        _Span(byte_boundaries[unit.chunk_begin], byte_boundaries[unit.chunk_end]) for unit in units
    ]


def _token_to_unit_mapping(
    text: str, units: tuple[_TextUnit, ...], piece_layout: _PieceLayout
) -> torch.Tensor:
    unit_spans = _unit_spans(text, units, piece_layout.coordinate_system)
    shape = (len(piece_layout.spans), len(unit_spans))
    if not piece_layout.spans or not unit_spans:
        return torch.zeros(shape, dtype=torch.float32)

    piece_begins = torch.tensor([span.begin for span in piece_layout.spans])[:, None]
    piece_ends = torch.tensor([span.end for span in piece_layout.spans])[:, None]
    unit_begins = torch.tensor([span.begin for span in unit_spans])[None, :]
    unit_ends = torch.tensor([span.end for span in unit_spans])[None, :]
    overlaps = (
        torch.minimum(piece_ends, unit_ends) - torch.maximum(piece_begins, unit_begins)
    ).clamp_min(0)
    overlaps = overlaps.to(torch.float32)
    totals = overlaps.sum(dim=1, keepdim=True)
    return torch.where(totals > 0, overlaps / totals.clamp_min(1), overlaps)


def _prepared_words(normalized_chunks: list[str]) -> list[_PreparedWord]:
    return [
        _PreparedWord(
            chunk_index=chunk_index,
            span=span,
            normalized=_normalized_word(chunk[span.begin : span.end]),
        )
        for chunk_index, chunk in enumerate(normalized_chunks)
        for span in _lexical_word_spans(chunk)
    ]


def _fast_map_source_words_to_chunks(
    source_words: list[_SourceWord], prepared_words: list[_PreparedWord], chunk_count: int
) -> list[list[_MappedWord]] | None:
    if len(source_words) != len(prepared_words):
        return None
    if any(
        _normalized_word(source_word.text) != prepared_word.normalized
        for source_word, prepared_word in zip(source_words, prepared_words)
    ):
        return None

    mapped: list[list[_MappedWord]] = [[] for _ in range(chunk_count)]
    for source_word, prepared_word in zip(source_words, prepared_words):
        mapped[prepared_word.chunk_index].append(_MappedWord(source_word, prepared_word.span))
    return mapped


def _best_effort_map_source_words_to_chunks(
    source_words: list[_SourceWord], prepared_words: list[_PreparedWord], chunk_count: int
) -> tuple[list[list[_MappedWord]], list[tuple[_Span, ...]], int]:
    """Map only unique monotonic word pairs so degraded timestamps never guess."""

    source_keys = [_normalized_word(word.text) for word in source_words]
    prepared_keys = [word.normalized for word in prepared_words]
    source_counts = Counter(source_keys)
    prepared_counts = Counter(prepared_keys)
    prepared_positions = {key: index for index, key in enumerate(prepared_keys)}

    pairs: list[tuple[int, int]] = []
    previous_prepared = -1
    for source_index, key in enumerate(source_keys):
        if source_counts[key] != 1 or prepared_counts[key] != 1:
            continue
        prepared_index = prepared_positions[key]
        if prepared_index <= previous_prepared:
            continue
        pairs.append((source_index, prepared_index))
        previous_prepared = prepared_index

    mapped: list[list[_MappedWord]] = [[] for _ in range(chunk_count)]
    paired_prepared = set()
    paired_source = set()
    for source_index, prepared_index in pairs:
        source_word = source_words[source_index]
        prepared_word = prepared_words[prepared_index]
        mapped[prepared_word.chunk_index].append(_MappedWord(source_word, prepared_word.span))
        paired_source.add(source_index)
        paired_prepared.add(prepared_index)

    ignored: list[list[_Span]] = [[] for _ in range(chunk_count)]
    for prepared_index, prepared_word in enumerate(prepared_words):
        if prepared_index not in paired_prepared:
            ignored[prepared_word.chunk_index].append(prepared_word.span)
    return mapped, [tuple(spans) for spans in ignored], len(source_words) - len(paired_source)


def _iter_timestamp_text_chunks(
    source_text: str,
    chunks: Iterable[str],
    sentencepiece_processor: object,
    *,
    best_effort: bool = False,
) -> Iterator[TimestampTextChunk]:
    """Lazily map prepared chunks, retaining strict behavior unless requested."""

    lexical_words = _lexical_words(source_text)
    normalized_chunks = [unicodedata.normalize("NFKC", chunk) for chunk in chunks]
    prepared_words = _prepared_words(normalized_chunks)
    mapped_words = _fast_map_source_words_to_chunks(
        lexical_words, prepared_words, len(normalized_chunks)
    )
    ignored_spans: list[tuple[_Span, ...]] = [() for _ in normalized_chunks]
    source_words = lexical_words

    if mapped_words is None:
        try:
            mapped_words = _map_source_words_to_chunks(
                source_text, normalized_chunks, lexical_words
            )
            source_words = [mapped.word for chunk_words in mapped_words for mapped in chunk_words]
        except ValueError:
            if not best_effort:
                raise
            mapped_words, ignored_spans, omitted_count = _best_effort_map_source_words_to_chunks(
                lexical_words, prepared_words, len(normalized_chunks)
            )
            logger.warning(
                "Timestamp text mapping omitted %d source word(s); audio generation continues",
                omitted_count,
            )

    piece_layout_reader = _SentencePieceLayoutReader(sentencepiece_processor)
    for normalized_chunk, chunk_words, chunk_ignored in zip(
        normalized_chunks, mapped_words, ignored_spans
    ):
        units = _build_units(
            source_text, source_words, normalized_chunk, chunk_words, chunk_ignored
        )
        piece_layout = piece_layout_reader.read(normalized_chunk)
        yield TimestampTextChunk(
            text=normalized_chunk,
            units=units,
            token_to_unit=_token_to_unit_mapping(normalized_chunk, units, piece_layout),
            prepared_tokens=torch.tensor(piece_layout.token_ids, dtype=torch.long)[None, :],
        )


def build_timestamp_text_chunks(
    source_text: str, chunks: Iterable[str], sentencepiece_processor: object
) -> list[TimestampTextChunk]:
    """Map prepared generation chunks back to exact lexical spans in source text."""

    return list(_iter_timestamp_text_chunks(source_text, chunks, sentencepiece_processor))
