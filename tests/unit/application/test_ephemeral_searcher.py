import pathlib
from unittest.mock import patch

import pytest
from rich.console import Console

from simgrep.core.models import OutputMode, SearchResult, SimgrepConfig
from simgrep.ephemeral_searcher import EphemeralSearcher


def test_ephemeral_searcher_show(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = SimgrepConfig(ephemeral_cache_dir=tmp_path / "cache")
    db = tmp_path / "meta.duckdb"
    idx = tmp_path / "index.usearch"
    with patch("simgrep.ephemeral_searcher.load_global_config", return_value=cfg), \
         patch("simgrep.ephemeral_searcher.get_ephemeral_cache_paths", return_value=(db, idx)), \
         patch("simgrep.ephemeral_searcher.Indexer"), \
         patch("simgrep.ephemeral_searcher.MetadataStore"), \
         patch("simgrep.ephemeral_searcher.SearchService") as mock_search_service:
        mock_search_service.return_value.search.return_value = [
            SearchResult(label=0, score=0.9, file_path=tmp_path / "file.txt", chunk_text="hello")
        ]
        console = Console(force_terminal=True, width=120)
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


def test_ephemeral_searcher_no_results(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = SimgrepConfig(ephemeral_cache_dir=tmp_path / "cache")
    db = tmp_path / "meta.duckdb"
    idx = tmp_path / "index.usearch"
    with patch("simgrep.ephemeral_searcher.load_global_config", return_value=cfg), \
         patch("simgrep.ephemeral_searcher.get_ephemeral_cache_paths", return_value=(db, idx)), \
         patch("simgrep.ephemeral_searcher.Indexer"), \
         patch("simgrep.ephemeral_searcher.MetadataStore"), \
         patch("simgrep.ephemeral_searcher.SearchService") as mock_search_service:
        mock_search_service.return_value.search.return_value = []
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
        assert "No relevant chunks" in out


def test_ephemeral_searcher_nonexistent_path(tmp_path: pathlib.Path) -> None:
    bad_path = tmp_path / "missing"
    console = Console(force_terminal=True, width=120)
    searcher = EphemeralSearcher(console=console)
    with pytest.raises(SystemExit):
        searcher.search(
            query_text="query",
            path_to_search=bad_path,
        )
