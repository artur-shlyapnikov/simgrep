import numpy as np
import pathlib
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

class DummyStore:
    def __init__(self, mapping):
        self.mapping = mapping
    def retrieve_chunk_details_persistent(self, label: int):
        return self.mapping.get(label)


def test_perform_persistent_search_show_results(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    console = Console()
    store = DummyStore({1: ("hello", pathlib.Path("a.txt"), 0, 5)})

    def fake_generate_embeddings(texts: list[str], model_name: str):
        return np.zeros((1, 3), dtype=np.float32)

    def fake_search_inmemory_index(index: object, query_embedding: np.ndarray, k: int = 5):
        return [(1, 0.95)]

    monkeypatch.setattr(searcher, "generate_embeddings", fake_generate_embeddings)
    monkeypatch.setattr(searcher, "search_inmemory_index", fake_search_inmemory_index)

    searcher.perform_persistent_search(
        query_text="hello",
        console=console,
        metadata_store=store,
        vector_index=object(),
        global_config=SimgrepConfig(),
        output_mode=OutputMode.show,
        min_score=0.5,
    )

    out = capsys.readouterr().out
    assert "Search Results" in out
    assert "hello" in out


def test_perform_persistent_search_paths_output(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    console = Console()
    store = DummyStore({1: ("t1", pathlib.Path("a.txt"), 0, 1), 2: ("t2", pathlib.Path("b.txt"), 1, 2)})

    def fake_generate_embeddings(texts: list[str], model_name: str):
        return np.zeros((1, 3), dtype=np.float32)

    def fake_search_inmemory_index(index: object, query_embedding: np.ndarray, k: int = 5):
        return [(1, 0.9), (2, 0.8), (1, 0.7)]

    monkeypatch.setattr(searcher, "generate_embeddings", fake_generate_embeddings)
    monkeypatch.setattr(searcher, "search_inmemory_index", fake_search_inmemory_index)

    searcher.perform_persistent_search(
        query_text="query",
        console=console,
        metadata_store=store,
        vector_index=object(),
        global_config=SimgrepConfig(),
        output_mode=OutputMode.paths,
        min_score=0.5,
    )

    out = capsys.readouterr().out
    assert "a.txt" in out
    assert "b.txt" in out
    assert out.strip().splitlines().count("a.txt") == 1


def test_perform_persistent_search_metadata_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    console = Console()

    class ErrorStore:
        def retrieve_chunk_details_persistent(self, label: int):
            raise searcher.MetadataDBError("boom")

    store = ErrorStore()

    def fake_generate_embeddings(texts: list[str], model_name: str):
        return np.zeros((1, 3), dtype=np.float32)

    def fake_search_inmemory_index(index: object, query_embedding: np.ndarray, k: int = 5):
        return [(1, 0.95)]

    monkeypatch.setattr(searcher, "generate_embeddings", fake_generate_embeddings)
    monkeypatch.setattr(searcher, "search_inmemory_index", fake_search_inmemory_index)

    searcher.perform_persistent_search(
        query_text="query",
        console=console,
        metadata_store=store,
        vector_index=object(),
        global_config=SimgrepConfig(),
        output_mode=OutputMode.show,
        min_score=0.5,
    )

    out = capsys.readouterr().out
    assert "Database error" in out
