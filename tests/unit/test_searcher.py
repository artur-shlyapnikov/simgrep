from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pytest
import usearch.index
from rich.console import Console

from simgrep import searcher
from simgrep.models import OutputMode, SearchResult, SimgrepConfig


class MinimalStore:
    def retrieve_filtered_chunk_details(
        self,
        usearch_labels: List[int],
        file_filter: Optional[List[str]] = None,
        keyword_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if 1 in usearch_labels:
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


def fake_generate_embeddings(texts: list[str], model_name: str, is_query: bool = False) -> np.ndarray:
    return np.zeros((1, 3), dtype=np.float32)


def fake_search_inmemory_index_low_score(index: object, query_embedding: np.ndarray, k: int = 5) -> List[SearchResult]:
    return [SearchResult(label=1, score=0.05)]


def fake_search_inmemory_index_no_results(index: object, query_embedding: np.ndarray, k: int = 5) -> List[SearchResult]:
    return []


@pytest.mark.parametrize(
    "search_fn, min_score, expected_message",
    [
        pytest.param(
            fake_search_inmemory_index_low_score,
            0.9,
            "No relevant chunks found in the persistent index (after filtering).",
            id="no_results_after_score_filter",
        ),
        pytest.param(
            fake_search_inmemory_index_no_results,
            0.1,
            "No relevant chunks found in the persistent index.",
            id="no_initial_results",
        ),
    ],
)
def test_perform_persistent_search_no_results(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    search_fn: Callable[..., List[SearchResult]],
    min_score: float,
    expected_message: str,
) -> None:
    """Tests that no results are shown, either from an empty search or filtering."""
    console = Console()
    store = MinimalStore()

    monkeypatch.setattr(searcher, "generate_embeddings", fake_generate_embeddings)
    monkeypatch.setattr(searcher, "search_inmemory_index", search_fn)

    vector_index = usearch.index.Index(ndim=3)
    # Add a dummy vector to avoid the `len(vector_index) == 0` check
    vector_index.add(np.array([0], dtype=np.int64), np.zeros((1, 3), dtype=np.float32))
    searcher.perform_persistent_search(
        query_text="irrelevant",
        console=console,
        metadata_store=store,  # type: ignore[arg-type]
        vector_index=vector_index,
        global_config=SimgrepConfig(),
        output_mode=OutputMode.show,
        min_score=min_score,
    )

    out = capsys.readouterr().out
    assert expected_message in out
