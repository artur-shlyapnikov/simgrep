import pathlib
import sys
from typing import Generator, List, Tuple

import duckdb
import pytest
import usearch.index
from rich.console import Console

from simgrep.config import DEFAULT_K_RESULTS, SimgrepConfig
from simgrep.indexer import Indexer, IndexerConfig
from simgrep.metadata_db import connect_persistent_db
from simgrep.models import OutputMode  # Assuming OutputMode is in models
from simgrep.searcher import perform_persistent_search
from simgrep.vector_store import load_persistent_index

# Minimal set of fixtures for this integration test


@pytest.fixture
def temp_project_dir_for_searcher(tmp_path: pathlib.Path) -> pathlib.Path:
    project_data_dir = tmp_path / "searcher_test_project_data"
    project_data_dir.mkdir(parents=True, exist_ok=True)
    return project_data_dir


@pytest.fixture
def global_config_for_searcher(
    temp_project_dir_for_searcher: pathlib.Path,
) -> SimgrepConfig:
    # Mock or create a SimgrepConfig that points its default_project_data_dir
    # to our temporary directory for this test.
    class MockSimgrepConfig(SimgrepConfig):
        default_project_data_dir: pathlib.Path = temp_project_dir_for_searcher

    cfg = MockSimgrepConfig()
    # Ensure the directory exists, though load_or_create_global_config usually does this.
    # For this fixture, we are directly controlling it.
    cfg.default_project_data_dir.mkdir(parents=True, exist_ok=True)
    return cfg


@pytest.fixture
def indexer_config_for_searcher(
    global_config_for_searcher: SimgrepConfig,
) -> IndexerConfig:
    return IndexerConfig(
        project_name="searcher_test_project",
        db_path=global_config_for_searcher.default_project_data_dir / "metadata.duckdb",
        usearch_index_path=global_config_for_searcher.default_project_data_dir
        / "index.usearch",
        embedding_model_name=global_config_for_searcher.default_embedding_model_name,
        chunk_size_tokens=global_config_for_searcher.default_chunk_size_tokens,
        chunk_overlap_tokens=global_config_for_searcher.default_chunk_overlap_tokens,
        file_scan_patterns=["*.txt"],
    )


@pytest.fixture
def sample_files_for_searcher(tmp_path: pathlib.Path) -> pathlib.Path:
    source_dir = tmp_path / "searcher_sample_files"
    source_dir.mkdir()
    (source_dir / "search_doc1.txt").write_text(
        "Semantic search is a key feature of simgrep. Simgrep helps find relevant information."
    )
    (source_dir / "search_doc2.txt").write_text(
        "Another document talking about vector databases and embeddings for simgrep."
    )
    (source_dir / "unrelated.txt").write_text(
        "This file contains kitchen recipes for apple pie."
    )
    return source_dir


@pytest.fixture
def populated_persistent_index_for_searcher(
    indexer_config_for_searcher: IndexerConfig,
    sample_files_for_searcher: pathlib.Path,
    test_console: Console,  # Reusing from test_indexer_persistent if available, or define one
    global_config_for_searcher: SimgrepConfig,  # Add this parameter
) -> Tuple[duckdb.DuckDBPyConnection, usearch.index.Index, SimgrepConfig]:
    """Creates a persistent index, then loads and returns its components."""
    indexer = Indexer(config=indexer_config_for_searcher, console=test_console)
    indexer.index_path(target_path=sample_files_for_searcher, wipe_existing=True)

    db_conn = connect_persistent_db(indexer_config_for_searcher.db_path)
    vector_idx = load_persistent_index(indexer_config_for_searcher.usearch_index_path)

    if vector_idx is None:
        pytest.fail("Failed to load vector index after populating.")

    yield db_conn, vector_idx, global_config_for_searcher  # Now global_config_for_searcher is a SimgrepConfig instance

    db_conn.close()


@pytest.fixture
def test_console() -> (
    Console
):  # Copied from test_indexer_persistent for self-containment if needed
    return Console(file=sys.stdout, width=120, quiet=True)


class TestSearcherPersistentIntegration:

    def test_perform_persistent_search_show_mode(
        self,
        populated_persistent_index_for_searcher: Tuple[
            duckdb.DuckDBPyConnection, usearch.index.Index, SimgrepConfig
        ],
        test_console: Console,
        capsys,  # To capture console output
    ):
        db_conn, vector_index, global_cfg = populated_persistent_index_for_searcher

        query = "simgrep information retrieval"

        perform_persistent_search(
            query_text=query,
            console=test_console,  # Use the quiet console for testing
            db_conn=db_conn,
            vector_index=vector_index,
            global_config=global_cfg,
            output_mode=OutputMode.show,
            k_results=2,
        )

        captured = capsys.readouterr()
        # Basic checks on output:
        assert "Embedding query" in captured.out
        assert "Searching persistent index" in captured.out
        assert "Search Results" in captured.out
        assert "File: " in captured.out  # Expecting at least one result
        assert "Score: " in captured.out
        assert "Chunk: " in captured.out
        # Check that it found relevant files
        assert "search_doc1.txt" in captured.out or "search_doc2.txt" in captured.out
        assert "unrelated.txt" not in captured.out  # Should be less relevant

    def test_perform_persistent_search_paths_mode(
        self,
        populated_persistent_index_for_searcher: Tuple[
            duckdb.DuckDBPyConnection, usearch.index.Index, SimgrepConfig
        ],
        test_console: Console,
        capsys,
        sample_files_for_searcher: pathlib.Path,  # To verify paths
    ):
        db_conn, vector_index, global_cfg = populated_persistent_index_for_searcher

        query = "vector databases"

        perform_persistent_search(
            query_text=query,
            console=test_console,
            db_conn=db_conn,
            vector_index=vector_index,
            global_config=global_cfg,
            output_mode=OutputMode.paths,
            k_results=3,
        )

        captured = capsys.readouterr()
        assert "Embedding query" in captured.out
        assert "Searching persistent index" in captured.out

        # Expected paths (absolute)
        expected_path1_abs = str(
            (sample_files_for_searcher / "search_doc2.txt").resolve()
        )

        assert expected_path1_abs in captured.out
        # search_doc1.txt might also appear depending on similarity and k
        # unrelated.txt should not appear
        assert "unrelated.txt" not in captured.out

    def test_perform_persistent_search_no_results(
        self,
        populated_persistent_index_for_searcher: Tuple[
            duckdb.DuckDBPyConnection, usearch.index.Index, SimgrepConfig
        ],
        test_console: Console,
        capsys,
    ):
        db_conn, vector_index, global_cfg = populated_persistent_index_for_searcher

        query = (
            "extremely obscure query with no matches expected zyxw"  # Unlikely to match
        )

        perform_persistent_search(
            query_text=query,
            console=test_console,
            db_conn=db_conn,
            vector_index=vector_index,
            global_config=global_cfg,
            output_mode=OutputMode.show,
            k_results=2,
        )

        captured = capsys.readouterr()
        assert "No relevant chunks found in the persistent index." in captured.out

    def test_perform_persistent_search_paths_mode_no_results(
        self,
        populated_persistent_index_for_searcher: Tuple[
            duckdb.DuckDBPyConnection, usearch.index.Index, SimgrepConfig
        ],
        test_console: Console,
        capsys,
    ):
        db_conn, vector_index, global_cfg = populated_persistent_index_for_searcher

        query = "extremely obscure query with no matches expected zyxw vuts"  # Unlikely to match

        perform_persistent_search(
            query_text=query,
            console=test_console,
            db_conn=db_conn,
            vector_index=vector_index,
            global_config=global_cfg,
            output_mode=OutputMode.paths,
            k_results=2,
        )

        captured = capsys.readouterr()
        assert "No matching files found." in captured.out

    # Add more tests:
    # - Different k_results
    # - Error handling (e.g., if DB connection fails mid-way, though harder to simulate here)
    # - Relative paths (though perform_persistent_search needs base_path_for_relativity properly set)
