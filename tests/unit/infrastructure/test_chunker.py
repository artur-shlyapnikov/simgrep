from __future__ import annotations

from dataclasses import dataclass

import pytest

from simgrep.adapters.chunker import HFChunker


@dataclass
class _Encoding:
    input_ids: list[int]
    offset_mapping: list[tuple[int, int]]


class _Tokenizer:
    def __init__(self, offsets: list[tuple[int, int]], *, model_max_length: int | None = None) -> None:
        self._offsets = offsets
        if model_max_length is not None:
            self.model_max_length = model_max_length

    def __call__(self, text: str, return_offsets_mapping: bool, add_special_tokens: bool, truncation: bool) -> _Encoding:
        del text, return_offsets_mapping, add_special_tokens, truncation
        return _Encoding(input_ids=list(range(len(self._offsets))), offset_mapping=self._offsets)


def _patch_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    offsets: list[tuple[int, int]],
    *,
    model_max_length: int | None = None,
) -> None:
    monkeypatch.setattr("simgrep.adapters.chunker.load_tokenizer", lambda _model_name: _Tokenizer(offsets, model_max_length=model_max_length))


def test_clamps_chunk_size_to_model_token_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "abcdef"
    offsets = [(i, i + 1) for i in range(len(text))]
    _patch_tokenizer(monkeypatch, offsets, model_max_length=5)

    chunker = HFChunker("fake-model", chunk_size=999, overlap=0)
    chunks = chunker.chunk(text)

    assert chunker.model_token_limit == 5
    assert chunker._chunk_size == 4
    assert all(chunk.tokens <= 4 for chunk in chunks)


def test_clamps_overlap_and_avoids_infinite_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "abcdef"
    offsets = [(i, i + 1) for i in range(len(text))]
    _patch_tokenizer(monkeypatch, offsets)

    chunker = HFChunker("fake-model", chunk_size=3, overlap=99)
    chunks = chunker.chunk(text)

    assert chunker._overlap == 2
    assert len(chunks) == len(text)


def test_overlap_creates_real_intersection_between_adjacent_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "abcdef"
    offsets = [(i, i + 1) for i in range(len(text))]
    _patch_tokenizer(monkeypatch, offsets)

    chunker = HFChunker("fake-model", chunk_size=4, overlap=2)
    chunks = chunker.chunk(text)

    assert len(chunks) >= 2
    assert chunks[0].text == "abcd"
    assert chunks[1].text == "cdef"


def test_chunk_offsets_and_text_mapping_are_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "one\ntwo\nthree\n"
    offsets = [(1, 2), (5, 6), (9, 10)]
    _patch_tokenizer(monkeypatch, offsets)

    chunks = HFChunker("fake-model", chunk_size=1, overlap=0).chunk(text)

    assert chunks
    for chunk in chunks:
        assert 0 <= chunk.start < chunk.end <= len(text)
        assert chunk.text == text[chunk.start : chunk.end]


def test_unicode_offsets_preserve_python_character_slicing(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "A🙂中B"
    offsets = [(0, 1), (1, 2), (2, 3), (3, 4)]
    _patch_tokenizer(monkeypatch, offsets)

    chunks = HFChunker("fake-model", chunk_size=2, overlap=1).chunk(text)

    assert chunks
    assert all(chunk.text == text[chunk.start : chunk.end] for chunk in chunks)
    assert any("🙂" in chunk.text for chunk in chunks)
    assert any("中" in chunk.text for chunk in chunks)


def test_drops_zero_width_and_bom_only_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "\ufeff\u200b\u200c\u200d"
    offsets = [(0, 1), (1, 2), (2, 3), (3, 4)]
    _patch_tokenizer(monkeypatch, offsets)

    chunks = HFChunker("fake-model", chunk_size=4, overlap=0).chunk(text)

    assert chunks == []


def test_line_expansion_respects_max_extra_chars(monkeypatch: pytest.MonkeyPatch) -> None:
    huge_line = "a" * 5000
    text = huge_line + "\n"
    offsets = [(2500, 2501)]
    _patch_tokenizer(monkeypatch, offsets)

    chunks = HFChunker("fake-model", chunk_size=1, overlap=0).chunk(text)

    assert len(chunks) == 1
    assert chunks[0].start == 2500
    assert chunks[0].end == 2501


def test_line_expansion_uses_full_line_when_within_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "first line\nsecond line\nthird line\n"
    second_start = text.index("second")
    offsets = [(second_start + 3, second_start + 4)]
    _patch_tokenizer(monkeypatch, offsets)

    chunks = HFChunker("fake-model", chunk_size=1, overlap=0).chunk(text)

    assert len(chunks) == 1
    assert chunks[0].text == "second line\n"
    assert chunks[0].start == second_start
    assert chunks[0].end == second_start + len("second line\n")


def test_very_long_single_line_does_not_expand_to_huge_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "x" * 20_000
    offsets = [(10, 11), (11, 12), (12, 13)]
    _patch_tokenizer(monkeypatch, offsets)

    chunks = HFChunker("fake-model", chunk_size=3, overlap=0).chunk(text)

    assert len(chunks) == 1
    assert chunks[0].text == text[10:13]


@pytest.mark.parametrize("text", ["", "   \n\t  "])
def test_empty_or_whitespace_only_text_returns_zero_chunks(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    _patch_tokenizer(monkeypatch, offsets=[])

    assert HFChunker("fake-model", chunk_size=8, overlap=2).chunk(text) == []
