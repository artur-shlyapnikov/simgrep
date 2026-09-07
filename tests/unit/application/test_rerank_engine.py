"""Application-level rerank orchestration over a FakeReranker (no real model)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simgrep.models import FileRole, SearchResult
from simgrep.rerank import rerank_results
from tests.conftest import FakeRuntime


def make_result(label: int, chunk_text: str) -> SearchResult:
    return SearchResult(
        label=label,
        score=0.5,
        file_path=Path(f"f{label}.py"),
        chunk_text=chunk_text,
        start_char=0,
        end_char=len(chunk_text),
        line_start=label,
        line_end=label,
        file_role=FileRole.source,
        language="python",
    )


class TestRerankThroughFakeReranker:
    def test_scores_are_cross_scores_and_order_desc(self, fake_runtime: FakeRuntime) -> None:
        # FakeReranker scores by len(doc) % 7 / 7 — craft distinct residues:
        # 6/7 > 3/7 > 1/7.
        results = [
            make_result(0, "x" * 1),  # 1/7
            make_result(1, "y" * 3),  # 3/7
            make_result(2, "z" * 6),  # 6/7
        ]
        reranker = fake_runtime.reranker
        outcome = rerank_results(results, "query", reranker.score, top=25)
        assert [r.label for r in outcome.results] == [2, 1, 0]
        assert outcome.results[0].score == pytest.approx(6 / 7)
        # The reranker saw the query and the window documents.
        assert len(reranker.calls) == 1
        seen_query, seen_docs = reranker.calls[0]
        assert seen_query == "query"
        assert seen_docs == ["x", "yyy", "zzzzzz"]

    def test_window_limits_what_the_model_sees(self, fake_runtime: FakeRuntime) -> None:
        results = [make_result(i, "a" * (i + 1)) for i in range(5)]
        rerank_results(results, "q", fake_runtime.reranker.score, top=2)
        _, docs = fake_runtime.reranker.calls[0]
        assert docs == ["a", "aa"]

    def test_runtime_reranker_defaults_to_fake_instance(self, fake_runtime: FakeRuntime) -> None:
        assert fake_runtime.reranker is not None
        scores = fake_runtime.reranker.score("q", ["abcd"])
        assert isinstance(scores, np.ndarray)
        assert scores.dtype == np.float32
