import pathlib
from typing import Any, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from rich.console import Console

from simgrep.core.models import OutputMode, SearchResult
from simgrep.ephemeral_searcher import EphemeralSearcher


@pytest.fixture
def mock_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mocks all external dependencies for EphemeralSearcher."""
    monkeypatch.setattr("simgrep.ephemeral_searcher.UnstructuredExtractor", MagicMock)
    monkeypatch.setattr("simgrep.ephemeral_searcher.HFTokenChunker", MagicMock)
    monkeypatch.setattr("simgrep.ephemeral_searcher.SentenceEmbedder", MagicMock)
    monkeypatch.setattr("simgrep.ephemeral_searcher.USearchIndex", MagicMock)
    monkeypatch.setattr("simgrep.ephemeral_searcher.MetadataStore", MagicMock)
    monkeypatch.setattr("simgrep.ephemeral_searcher.SearchService", MagicMock)
    monkeypatch.setattr("simgrep.ephemeral_searcher.gather_files_to_process", MagicMock)


def test_ephemeral_searcher_show(mock_dependencies: None, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    test_file = tmp_path / "file.txt"
    test_file.write_text("hello")

    with patch("simgrep.ephemeral_searcher.gather_files_to_process") as mock_gather:
        mock_gather.return_value = [test_file]
        with patch("simgrep.ephemeral_searcher.SearchService") as mock_search_service:
            mock_instance = mock_search_service.return_value
            mock_instance.search.return_value = [
                {
                    "file_path": test_file,
                    "chunk_text": "hello",
                    "score": 0.99,
                }
            ]

            console = Console(force_terminal=True, width=120)  # Force terminal for consistent output
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


def test_ephemeral_searcher_no_results(mock_dependencies: None, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    test_file = tmp_path / "file.txt"
    test_file.write_text("hello")

    with patch("simgrep.ephemeral_searcher.gather_files_to_process") as mock_gather:
        mock_gather.return_value = [test_file]
        with patch("simgrep.ephemeral_searcher.SearchService") as mock_search_service:
            mock_instance = mock_search_service.return_value
            mock_instance.search.return_value = []  # No results

            console = Console(force_terminal=True, width=120)
            searcher = EphemeralSearcher(console=console)
            searcher.search(
                query_text="goodbye",
                path_to_search=tmp_path,
                patterns=["*.txt"],
                output_mode=OutputMode.show,
                top=1,
            )

            out = capsys.readouterr().out
            assert "No relevant chunks found" in out
