import pathlib
from typing import Any, Dict, List

import numpy as np
import pytest
from rich.console import Console

from simgrep.ephemeral_searcher import EphemeralSearcher
from simgrep.models import OutputMode, SearchResult


class DummyTokenizer:
    pass


class DummyModel:
    pass


def fake_load_tokenizer(model_name: str) -> DummyTokenizer:
    return DummyTokenizer()


def fake_extract_text_from_file(path: pathlib.Path) -> str:
    return "dummy text for testing"


def fake_chunk_text_by_tokens(**kwargs: Any) -> List[Dict[str, Any]]:
    return [
        {
            "text": "chunk text",
            "start_char_offset": 0,
            "end_char_offset": 5,
            "token_count": 1,
        }
    ]


def fake_load_embedding_model(name: str) -> DummyModel:
    return DummyModel()


def fake_generate_embeddings(texts: List[str], model_name: str, model: DummyModel | None = None, is_query: bool = False) -> np.ndarray:
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    if is_query:
        return vec.reshape(1, -1)
    return np.tile(vec, (len(texts), 1))


def fake_create_inmemory_index(**kwargs: Any) -> object:
    return object()


def fake_search_inmemory_index(**kwargs: Any) -> List[SearchResult]:
    return [SearchResult(label=0, score=0.9)]


@pytest.fixture
def monkeypatched_searcher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "simgrep.ephemeral_searcher.load_tokenizer",
        fake_load_tokenizer,
    )
    monkeypatch.setattr(
        "simgrep.ephemeral_searcher.extract_text_from_file",
        fake_extract_text_from_file,
    )
    monkeypatch.setattr(
        "simgrep.ephemeral_searcher.chunk_text_by_tokens",
        fake_chunk_text_by_tokens,
    )
    monkeypatch.setattr(
        "simgrep.ephemeral_searcher.load_embedding_model",
        fake_load_embedding_model,
    )
    monkeypatch.setattr(
        "simgrep.ephemeral_searcher.generate_embeddings",
        fake_generate_embeddings,
    )
    monkeypatch.setattr(
        "simgrep.ephemeral_searcher.create_inmemory_index",
        fake_create_inmemory_index,
    )
    monkeypatch.setattr(
        "simgrep.ephemeral_searcher.search_inmemory_index",
        fake_search_inmemory_index,
    )


def test_ephemeral_searcher_show(monkeypatched_searcher: None, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    test_file = tmp_path / "file.txt"
    test_file.write_text("hello")

    console = Console()
    searcher = EphemeralSearcher(console=console)
    searcher.search(
        query_text="hello",
        path_to_search=tmp_path,
        patterns=["*.txt"],
        output_mode=OutputMode.show,
        top=1,
    )

    out = capsys.readouterr().out
    assert "Search Results" in out
    assert "file.txt" in out


def test_ephemeral_searcher_no_results(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    test_file = tmp_path / "file.txt"
    test_file.write_text("hello")

    monkeypatch.setattr("simgrep.ephemeral_searcher.search_inmemory_index", lambda **_: [])
    monkeypatch.setattr(
        "simgrep.ephemeral_searcher.load_tokenizer",
        fake_load_tokenizer,
    )
    monkeypatch.setattr(
        "simgrep.ephemeral_searcher.extract_text_from_file",
        fake_extract_text_from_file,
    )
    monkeypatch.setattr(
        "simgrep.ephemeral_searcher.chunk_text_by_tokens",
        fake_chunk_text_by_tokens,
    )
    monkeypatch.setattr(
        "simgrep.ephemeral_searcher.load_embedding_model",
        fake_load_embedding_model,
    )
    monkeypatch.setattr(
        "simgrep.ephemeral_searcher.generate_embeddings",
        fake_generate_embeddings,
    )
    monkeypatch.setattr(
        "simgrep.ephemeral_searcher.create_inmemory_index",
        fake_create_inmemory_index,
    )

    console = Console()
    searcher = EphemeralSearcher(console=console)
    searcher.search(
        query_text="hello",
        path_to_search=tmp_path,
        patterns=["*.txt"],
        output_mode=OutputMode.show,
        top=1,
    )

    out = capsys.readouterr().out
    assert "No relevant chunks found" in out


def test_ephemeral_searcher_quiet(monkeypatched_searcher: None, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    test_file = tmp_path / "file.txt"
    test_file.write_text("hello")

    console = Console()
    searcher = EphemeralSearcher(console=console, quiet=True)
    searcher.search(
        query_text="hello",
        path_to_search=tmp_path,
        patterns=["*.txt"],
        output_mode=OutputMode.show,
        top=1,
    )

    out = capsys.readouterr().out
    assert "Search Results" in out
    assert "Processing files" not in out
