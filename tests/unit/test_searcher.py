from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytest
import usearch.index
from rich.console import Console

from simgrep import searcher
from simgrep.models import OutputMode, SearchResult, SimgrepConfig


class MinimalStore:
    def retrieve_chunk_details_persistent(self, label: int) -> Optional[Tuple[str, Path, int, int]]:
        return None


def fake_generate_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    return np.zeros((1, 3), dtype=np.float32)


def fake_search_inmemory_index(index: object, query_embedding: np.ndarray, k: int = 5) -> List[SearchResult]:
    return [SearchResult(label=1, score=0.05)]


def test_perform_persistent_search_no_results(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    console = Console()
    store = MinimalStore()

    monkeypatch.setattr(searcher, "generate_embeddings", fake_generate_embeddings)
    monkeypatch.setattr(searcher, "search_inmemory_index", fake_search_inmemory_index)

    searcher.perform_persistent_search(
        query_text="irrelevant",
        console=console,
        metadata_store=store,  # type: ignore[arg-type]
        vector_index=usearch.index.Index(ndim=3),
        global_config=SimgrepConfig(),
        output_mode=OutputMode.show,
        min_score=0.9,
    )

    out = capsys.readouterr().out
    assert "No relevant chunks found in the persistent index." in out