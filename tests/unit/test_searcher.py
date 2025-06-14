from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pytest
import usearch.index
from rich.console import Console

from simgrep.adapters.sentence_embedder import SentenceEmbedder
from simgrep.adapters.usearch_index import USearchIndex
from simgrep.core.models import OutputMode, SearchResult, SimgrepConfig
from simgrep.services.search_service import SearchService


class MinimalStore:
    def retrieve_filtered_chunk_details(
        self,
        usearch_labels: List[int],
        file_filter: Optional[List[str]] = None,
        keyword_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # Simulate filtering: only return details for label 1
        if 1 in usearch_labels:
            # Simulate keyword filter not matching
            if keyword_filter and "some text" not in keyword_filter:
                return []
            # Simulate file filter not matching
            if file_filter and not any(Path("/fake/file.txt").match(p) for p in file_filter):
                return []

            return [
                {
                    "file_path": Path("/fake/file.txt"),
                    "chunk_text": "some text",
                    "start_char_offset": 0,
                    "end_char_offset": 9,
                    "usearch_label": 1,
                }
            ]
        return []

    def close(self) -> None:
        pass


def fake_search_low_score(self, vec: np.ndarray, k: int) -> List[SearchResult]:
    return [SearchResult(label=1, score=0.05)]


def fake_search_no_results(self, vec: np.ndarray, k: int) -> List[SearchResult]:
    return []


@pytest.mark.parametrize(
    "search_fn, min_score, expected_results_len",
    [
        pytest.param(
            fake_search_low_score,
            0.9,
            0,
            id="no_results_after_score_filter",
        ),
        pytest.param(
            fake_search_no_results,
            0.1,
            0,
            id="no_initial_results",
        ),
    ],
)
def test_search_service_no_results(
    monkeypatch: pytest.MonkeyPatch,
    search_fn: Callable[..., List[SearchResult]],
    min_score: float,
    expected_results_len: int,
) -> None:
    """Tests that SearchService returns no results, either from an empty search or filtering."""
    store = MinimalStore()
    embedder = SentenceEmbedder(model_name="Qwen/Qwen3-Embedding-0.6B")
    vector_index = USearchIndex(ndim=embedder.ndim)
    vector_index.add(np.array([0], dtype=np.int64), np.zeros((1, embedder.ndim), dtype=np.float32))

    # Monkeypatch the search method on the instance
    monkeypatch.setattr(USearchIndex, "search", search_fn)

    service = SearchService(store=store, embedder=embedder, index=vector_index)  # type: ignore

    results = service.search(
        query="irrelevant",
        k=5,
        min_score=min_score,
        file_filter=None,
        keyword_filter=None,
    )

    assert len(results) == expected_results_len
