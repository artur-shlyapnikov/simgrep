import pathlib
from unittest.mock import patch

import pytest
from rich.console import Console

from simgrep.core.abstractions import Embedder, TextExtractor, TokenChunker
from simgrep.core.context import SimgrepContext
from simgrep.core.models import OutputMode, SearchResult
from simgrep.ephemeral_searcher import EphemeralSearcher


@pytest.fixture
def fake_context(
    fake_text_extractor: TextExtractor,
    fake_token_chunker: TokenChunker,
    fake_embedder: Embedder,
    fake_vector_index_factory,
) -> SimgrepContext:
    return SimgrepContext(
        extractor=fake_text_extractor,
        chunker=fake_token_chunker,
        embedder=fake_embedder,
        index_factory=lambda ndim: fake_vector_index_factory(ndim=ndim),
    )


def test_ephemeral_searcher_show(fake_context: SimgrepContext, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    test_file = tmp_path / "file.txt"
    test_file.write_text("hello")

    with patch("simgrep.ephemeral_searcher.gather_files_to_process") as mock_gather:
        mock_gather.return_value = [test_file]
        with patch("simgrep.ephemeral_searcher.SearchService") as mock_search_service:
            mock_instance = mock_search_service.return_value
            mock_instance.search.return_value = [SearchResult(label=0, score=0.99, file_path=test_file, chunk_text="hello")]

            console = Console(force_terminal=True, width=120)
            searcher = EphemeralSearcher(context=fake_context, console=console)
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


def test_ephemeral_searcher_no_results(fake_context: SimgrepContext, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    test_file = tmp_path / "file.txt"
    test_file.write_text("hello")

    with patch("simgrep.ephemeral_searcher.gather_files_to_process") as mock_gather:
        mock_gather.return_value = [test_file]
        with patch("simgrep.ephemeral_searcher.SearchService") as mock_search_service:
            mock_instance = mock_search_service.return_value
            mock_instance.search.return_value = []  # No results

            console = Console(force_terminal=True, width=120)
            searcher = EphemeralSearcher(context=fake_context, console=console)
            searcher.search(
                query_text="goodbye",
                path_to_search=tmp_path,
                patterns=["*.txt"],
                output_mode=OutputMode.show,
                top=1,
            )

            out = capsys.readouterr().out
            assert "No relevant chunks found" in out
