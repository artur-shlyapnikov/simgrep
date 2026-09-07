"""Pure rerank math: orderings, windowed result reranking, best-per-file."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from simgrep.errors import RerankError
from simgrep.models import Chunk, FileRole, RerankMatch, SearchResult
from simgrep.rerank import best_per_file, chunk_file_texts, ensure_chunk_cap, rerank_orderings, rerank_results


def make_result(label: int, chunk_text: str, score: float = 0.5) -> SearchResult:
    return SearchResult(
        label=label,
        score=score,
        file_path=Path(f"f{label}.py"),
        chunk_text=chunk_text,
        start_char=0,
        end_char=len(chunk_text),
        line_start=label,
        line_end=label,
        file_role=FileRole.source,
        language="python",
    )


class TestRerankOrderings:
    def test_descending_order(self) -> None:
        assert rerank_orderings([0.1, 0.9, 0.5]) == (1, 2, 0)

    def test_stable_ties_keep_original_index(self) -> None:
        assert rerank_orderings([0.5, 0.9, 0.5]) == (1, 0, 2)

    def test_empty(self) -> None:
        assert rerank_orderings([]) == ()

    def test_single(self) -> None:
        assert rerank_orderings([-3.0]) == (0,)


class TestRerankResults:
    def test_window_cutover_preserves_tail(self) -> None:
        results = [make_result(i, f"chunk{i}") for i in range(5)]
        # Cross scores only defined for the queried window; first two entries.
        cross = {0: 0.2, 1: 0.8}

        def score_fn(query: str, docs: list[str]) -> np.ndarray:
            del query
            return np.asarray([cross[int(d[5:])] for d in docs], dtype=np.float32)

        outcome = rerank_results(results, "q", score_fn, top=2)
        assert [r.label for r in outcome.results] == [1, 0, 2, 3, 4]
        # Score field := cross score inside the window; tail untouched.
        assert outcome.results[0].score == pytest.approx(0.8)
        assert outcome.results[1].score == pytest.approx(0.2)
        assert [r.score for r in outcome.results[2:]] == [0.5, 0.5, 0.5]

    def test_top_ge_len_reorders_everything(self) -> None:
        results = [make_result(i, "c" * (i + 1)) for i in range(3)]

        def score_fn(query: str, docs: list[str]) -> np.ndarray:
            del query
            return np.asarray([float(len(d)) for d in docs], dtype=np.float32)

        outcome = rerank_results(results, "q", score_fn, top=10)
        assert [r.label for r in outcome.results] == [2, 1, 0]

    def test_empty_input(self) -> None:
        outcome = rerank_results([], "q", lambda q, d: np.zeros(len(d)), top=25)
        assert outcome.results == []

    def test_score_replacement_uses_cross_not_hybrid(self) -> None:
        results = [make_result(0, "a", score=0.01), make_result(1, "bb", score=0.99)]

        def score_fn(query: str, docs: list[str]) -> np.ndarray:
            del query
            return np.asarray([7.0, -1.0], dtype=np.float32)

        outcome = rerank_results(results, "q", score_fn, top=2)
        assert outcome.results[0].label == 0
        assert outcome.results[0].score == pytest.approx(7.0)
        assert outcome.results[1].score == pytest.approx(-1.0)

    def test_tie_break_by_original_index_within_window(self) -> None:
        results = [make_result(i, f"c{i}") for i in range(4)]
        outcome = rerank_results(results, "q", lambda q, d: np.ones(len(d), dtype=np.float32), top=3)
        # All-equal cross scores: window keeps incoming order.
        assert [r.label for r in outcome.results] == [0, 1, 2, 3]


def match(file_path: str, line_start: int, score: float) -> RerankMatch:
    return RerankMatch(
        file_path=file_path,
        line_start=line_start,
        line_end=line_start + 3,
        score=score,
        snippet="s" * 10,
    )


class TestBestPerFile:
    def test_dedupe_keeps_max_score(self) -> None:
        matches = (match("a.py", 1, 0.3), match("b.py", 5, 0.9), match("a.py", 40, 0.7))
        best = best_per_file(matches)
        assert [(m.file_path, m.line_start, m.score) for m in best] == [
            ("b.py", 5, 0.9),
            ("a.py", 40, 0.7),
        ]

    def test_tie_break_line_start_asc(self) -> None:
        matches = (match("a.py", 50, 0.6), match("a.py", 10, 0.6))
        best = best_per_file(matches)
        assert len(best) == 1
        assert best[0].line_start == 10

    def test_output_descending_by_score(self) -> None:
        matches = (
            match("c.py", 1, 0.1),
            match("a.py", 1, 0.5),
            match("b.py", 1, 0.9),
        )
        best = best_per_file(matches)
        assert [m.file_path for m in best] == ["b.py", "a.py", "c.py"]

    def test_empty(self) -> None:
        assert best_per_file(()) == ()


class TestEnsureChunkCap:
    def test_under_cap_passes(self) -> None:
        ensure_chunk_cap(511, 512)
        ensure_chunk_cap(512, 512)

    def test_over_cap_raises_rerank_error(self) -> None:
        with pytest.raises(RerankError):
            ensure_chunk_cap(513, 512)


class TestChunkFileTexts:
    def test_pass_through_preserves_chunks(self) -> None:
        chunks = [
            Chunk(id=7, file_id=3, text="alpha", start=0, end=5, tokens=1, line_start=1, line_end=2),
            Chunk(id=8, file_id=3, text="beta", start=6, end=10, tokens=1, line_start=4, line_end=4),
        ]

        class StubChunker:
            def chunk(self, text: str) -> Sequence[Chunk]:
                del text
                return chunks

        out = chunk_file_texts("ignored", StubChunker())
        assert isinstance(out, list)
        assert out == chunks
        assert [c.line_start for c in out] == [1, 4]
        assert [c.line_end for c in out] == [2, 4]
