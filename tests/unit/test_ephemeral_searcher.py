import pathlib
from typing import Any, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from rich.console import Console

from simgrep.core.context import SimgrepContext
from simgrep.core.models import Chunk, OutputMode, SearchResult
from simgrep.ephemeral_searcher import EphemeralSearcher


@pytest.fixture
def mock_context() -> SimgrepContext:
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = "hello"

    mock_chunker = MagicMock()
    mock_chunker.chunk.return_value = [Chunk(id=-1, file_id=-1, text="hello", start=0, end=5, tokens=1)]

    mock_embedder = MagicMock()
    mock_embedder.ndim = 3
    mock_embedder.encode.return_value = np.zeros((1, 3))

    mock_index = MagicMock()
    mock_index_factory = MagicMock(return_value=mock_index)

    return SimgrepContext(
        extractor=mock_extractor,
        chunker=mock_chunker,
        embedder=mock_embedder,
        index_factory=mock_index_factory,
    )


def test_ephemeral_searcher_show(mock_context: SimgrepContext, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    test_file = tmp_path / "file.txt"
    test_file.write_text("hello")

    with patch("simgrep.ephemeral_searcher.gather_files_to_process") as mock_gather:
        mock_gather.return_value = [test_file]
        with patch("simgrep.ephemeral_searcher.SearchService") as mock_search_service:
            mock_instance = mock_search_service.return_value
            mock_instance.search.return_value = [SearchResult(label=0, score=0.99, file_path=test_file, chunk_text="hello")]

            console = Console(force_terminal=True, width=120)
            searcher = EphemeralSearcher(context=mock_context, console=console)
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


def test_ephemeral_searcher_no_results(mock_context: SimgrepContext, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    test_file = tmp_path / "file.txt"
    test_file.write_text("hello")

    with patch("simgrep.ephemeral_searcher.gather_files_to_process") as mock_gather:
        mock_gather.return_value = [test_file]
        with patch("simgrep.ephemeral_searcher.SearchService") as mock_search_service:
            mock_instance = mock_search_service.return_value
            mock_instance.search.return_value = []  # No results

            console = Console(force_terminal=True, width=120)
            searcher = EphemeralSearcher(context=mock_context, console=console)
            searcher.search(
                query_text="goodbye",
                path_to_search=tmp_path,
                patterns=["*.txt"],
                output_mode=OutputMode.show,
                top=1,
            )

            out = capsys.readouterr().out
            assert "No relevant chunks found" in out