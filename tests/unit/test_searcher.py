import numpy as np
import pytest
from rich.console import Console

from simgrep import searcher
from simgrep.models import OutputMode, SimgrepConfig


class MinimalStore:
    def retrieve_chunk_details_persistent(self, label: int):
        return None


def test_perform_persistent_search_no_results(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    console = Console()
    store = MinimalStore()

    def fake_generate_embeddings(texts: list[str], model_name: str):
        return np.zeros((1, 3), dtype=np.float32)

    def fake_search_inmemory_index(index: object, query_embedding: np.ndarray, k: int = 5):
        return [(1, 0.05)]

    monkeypatch.setattr(searcher, "generate_embeddings", fake_generate_embeddings)
    monkeypatch.setattr(searcher, "search_inmemory_index", fake_search_inmemory_index)

    searcher.perform_persistent_search(
        query_text="irrelevant",
        console=console,
        metadata_store=store,
        vector_index=object(),
        global_config=SimgrepConfig(),
        output_mode=OutputMode.show,
        min_score=0.9,
    )

    out = capsys.readouterr().out
    assert "No relevant chunks found in the persistent index." in out
