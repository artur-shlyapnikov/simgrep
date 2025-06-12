import pathlib
from typing import Any, List

import pytest
from rich.console import Console

from simgrep.ephemeral_searcher import EphemeralSearcher
from simgrep.models import OutputMode


class DummyStore:
    def __init__(self, persistent: bool = False, db_path: pathlib.Path | None = None) -> None:
        self.path = db_path
    def close(self) -> None:
        pass


class DummyIndexer:
    calls: List[str] = []

    def __init__(self, config: Any, console: Console) -> None:
        self.config = config
        self.console = console
        DummyIndexer.calls.append("init")

    def run_index(self, paths: List[pathlib.Path], wipe_existing: bool) -> None:
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.db_path.write_text("db")
        self.config.usearch_index_path.write_text("idx")
        DummyIndexer.calls.append("run_index")


def dummy_load_index(path: pathlib.Path) -> object:
    dummy_load_index.calls += 1
    return object()

dummy_load_index.calls = 0


def dummy_perform_search(**kwargs: Any) -> None:
    dummy_perform_search.calls += 1

dummy_perform_search.calls = 0


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("simgrep.ephemeral_searcher.Indexer", DummyIndexer)
    monkeypatch.setattr("simgrep.ephemeral_searcher.MetadataStore", DummyStore)
    monkeypatch.setattr("simgrep.ephemeral_searcher.load_persistent_index", dummy_load_index)
    monkeypatch.setattr("simgrep.ephemeral_searcher.perform_persistent_search", dummy_perform_search)


def test_ephemeral_searcher_caches(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "file.txt"
    src.write_text("hello")

    console = Console()
    searcher = EphemeralSearcher(console=console)

    searcher.search("hello", tmp_path, patterns=["*.txt"], output_mode=OutputMode.json)
    assert "run_index" in DummyIndexer.calls
    first_calls = dummy_load_index.calls

    DummyIndexer.calls.clear()
    searcher.search("hello", tmp_path, patterns=["*.txt"], output_mode=OutputMode.json)
    assert "run_index" not in DummyIndexer.calls
    assert dummy_load_index.calls == first_calls + 1
    assert dummy_perform_search.calls == 2
