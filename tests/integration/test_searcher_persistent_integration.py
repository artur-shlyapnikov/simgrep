from pathlib import Path
from typing import Generator, Tuple

import pytest
import usearch.index
from rich.console import Console

from simgrep.indexer import Indexer, IndexerConfig
from simgrep.metadata_store import MetadataStore
from simgrep.models import OutputMode, SimgrepConfig
from simgrep.searcher import perform_persistent_search
from simgrep.vector_store import load_persistent_index

pytest.importorskip("transformers")
pytest.importorskip("sentence_transformers")
pytest.importorskip("usearch.index")


# Fixtures
@pytest.fixture
def test_console() -> Console:
    """Provides a Rich Console instance that writes to stdout, compatible with capsys."""
    return Console(width=120)


@pytest.fixture(scope="session")
def persistent_search_test_data_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Creates dummy files in a temporary directory for indexing in persistent search tests."""
    data_dir = tmp_path_factory.mktemp("persistent_search_data")

    file1_content = "This file talks about simgrep and advanced information retrieval techniques."
    (data_dir / "file1.txt").write_text(file1_content)

    file2_content = "Another document mentioning simgrep. Semantic search is powerful."
    (data_dir / "file2.txt").write_text(file2_content)

    file3_content = "A completely unrelated file about apples and oranges."
    (data_dir / "file3.txt").write_text(file3_content)

    return data_dir


@pytest.fixture(scope="session")
def default_simgrep_config_for_search_tests(
    tmp_path_factory: pytest.TempPathFactory,
) -> SimgrepConfig:
    """Provides a SimgrepConfig with a temporary data directory for persistent search tests."""
    simgrep_root_config_dir = tmp_path_factory.mktemp("simgrep_config_root_searcher")
    default_proj_data_dir = simgrep_root_config_dir / "default_project"
    # SimgrepConfig's default_project_data_dir.mkdir is called by load_or_create_global_config.
    # Here, we explicitly create it for the fixture's SimgrepConfig instance.
    default_proj_data_dir.mkdir(parents=True, exist_ok=True)

    cfg = SimgrepConfig(default_project_data_dir=default_proj_data_dir)
    return cfg


@pytest.fixture(scope="session")
def populated_persistent_index_for_searcher(
    persistent_search_test_data_path: Path,
    default_simgrep_config_for_search_tests: SimgrepConfig,
) -> Generator[Tuple[MetadataStore, usearch.index.Index, SimgrepConfig], None, None]:
    """
    Indexes dummy data and provides the DB connection, USearch index, and config for tests.
    This is session-scoped for efficiency as indexing can be slow.
    """
    cfg = default_simgrep_config_for_search_tests

    # Define unique paths for this fixture's index to avoid conflicts if other fixtures use the same dir
    db_file = cfg.default_project_data_dir / "test_searcher_fixture_metadata.duckdb"
    usearch_file = cfg.default_project_data_dir / "test_searcher_fixture_index.usearch"

    # Clean up previous run's files if they exist, to ensure a fresh index for the session
    if db_file.exists():
        db_file.unlink()
    if usearch_file.exists():
        usearch_file.unlink()

    indexer_config = IndexerConfig(
        project_name="test_searcher_fixture_project",
        db_path=db_file,
        usearch_index_path=usearch_file,
        embedding_model_name=cfg.default_embedding_model_name,
        chunk_size_tokens=cfg.default_chunk_size_tokens,
        chunk_overlap_tokens=cfg.default_chunk_overlap_tokens,
        file_scan_patterns=["*.txt"],
    )

    # Use a quiet console for the indexer to avoid polluting test logs
    indexer_console = Console(quiet=True)
    indexer = Indexer(config=indexer_config, console=indexer_console)

    indexer.run_index(target_paths=[persistent_search_test_data_path], wipe_existing=True)

    store = MetadataStore(persistent=True, db_path=db_file)
    vector_index = load_persistent_index(usearch_file)

    if vector_index is None:
        pytest.fail("Failed to load persistent vector index in 'populated_persistent_index_for_searcher' fixture.")

    yield store, vector_index, cfg

    # Teardown: close DB connection. Temp files are handled by tmp_path_factory.
    store.close()
    # Optional: clean up the specific index files if desired, though tmp_path_factory handles the root temp dir.
    # if db_file.exists(): db_file.unlink()
    # if usearch_file.exists(): usearch_file.unlink()


class TestSearcherPersistentIntegration:
    def test_perform_persistent_search_show_mode_with_results(
        self,
        populated_persistent_index_for_searcher: Tuple[MetadataStore, usearch.index.Index, SimgrepConfig],
        test_console: Console,
        capsys: pytest.CaptureFixture,
        persistent_search_test_data_path: Path,
    ) -> None:
        store, vector_index_val, global_cfg = populated_persistent_index_for_searcher
        query = "simgrep information retrieval"  # Should match content in file1.txt

        perform_persistent_search(
            query_text=query,
            console=test_console,
            metadata_store=store,
            vector_index=vector_index_val,
            global_config=global_cfg,
            output_mode=OutputMode.show,
            k_results=2,
            min_score=0.25,  # Adjusted min_score
        )
        captured = capsys.readouterr()

        assert "Embedding query" in captured.out, "Initial 'Embedding query' print message missing."

        # Robust check for file path
        expected_path_str = str(persistent_search_test_data_path / "file1.txt")
        cleaned_output = captured.out.replace("\n", "").replace("\r", "")
        assert f"File: {expected_path_str}" in cleaned_output, f"Expected 'File: {expected_path_str}' to be among the results."

        assert "Score:" in captured.out, "Output for 'show' mode should contain 'Score:'."
        assert "Chunk:" in captured.out, "Output for 'show' mode should contain 'Chunk:'."

    def test_perform_persistent_search_paths_mode_with_results(
        self,
        populated_persistent_index_for_searcher: Tuple[MetadataStore, usearch.index.Index, SimgrepConfig],
        test_console: Console,
        capsys: pytest.CaptureFixture,
        persistent_search_test_data_path: Path,
    ) -> None:
        store, vector_index_val, global_cfg = populated_persistent_index_for_searcher
        query = "semantic search"  # Should match content in file2.txt

        perform_persistent_search(
            query_text=query,
            console=test_console,
            metadata_store=store,
            vector_index=vector_index_val,
            global_config=global_cfg,
            output_mode=OutputMode.paths,
            k_results=2,
            min_score=0.25,  # Adjusted min_score
        )
        captured = capsys.readouterr()
        assert "Embedding query" in captured.out, "Initial 'Embedding query' print message missing."

        # Robust check for file path
        expected_path_str = str(persistent_search_test_data_path / "file2.txt")
        cleaned_output = captured.out.replace("\n", "").replace("\r", "")
        assert expected_path_str in cleaned_output, f"Expected '{expected_path_str}' in paths output."

    def test_perform_persistent_search_show_mode_no_results(
        self,
        populated_persistent_index_for_searcher: Tuple[MetadataStore, usearch.index.Index, SimgrepConfig],
        test_console: Console,
        capsys: pytest.CaptureFixture,
    ) -> None:
        store, vector_index_val, global_cfg = populated_persistent_index_for_searcher
        query = "zzxxyy_non_existent_term_qwerty_12345"  # Highly unlikely to match

        perform_persistent_search(
            query_text=query,
            console=test_console,
            metadata_store=store,
            vector_index=vector_index_val,
            global_config=global_cfg,
            output_mode=OutputMode.show,
            k_results=2,
            min_score=0.95,
        )
        captured = capsys.readouterr()
        assert "Embedding query" in captured.out, "Initial 'Embedding query' print message missing."
        assert (
            "No relevant chunks found in the persistent index (after filtering)." in captured.out
        ), "Expected 'no results' message for show mode."

    def test_perform_persistent_search_paths_mode_no_results(
        self,
        populated_persistent_index_for_searcher: Tuple[MetadataStore, usearch.index.Index, SimgrepConfig],
        test_console: Console,
        capsys: pytest.CaptureFixture,
    ) -> None:
        store, vector_index_val, global_cfg = populated_persistent_index_for_searcher
        query = "zzxxyy_non_existent_term_qwerty_12345"  # Highly unlikely to match

        perform_persistent_search(
            query_text=query,
            console=test_console,
            metadata_store=store,
            vector_index=vector_index_val,
            global_config=global_cfg,
            output_mode=OutputMode.paths,
            k_results=2,
            min_score=0.95,
        )
        captured = capsys.readouterr()
        assert "Embedding query" in captured.out, "Initial 'Embedding query' print message missing."
        assert "No matching files found." in captured.out, "Expected 'no results' message for paths mode."

    def test_perform_persistent_search_respects_k_results(
        self,
        populated_persistent_index_for_searcher: Tuple[
            MetadataStore,
            usearch.index.Index,
            SimgrepConfig,
        ],
        test_console: Console,
        capsys: pytest.CaptureFixture,
    ) -> None:
        store, vector_index_val, global_cfg = populated_persistent_index_for_searcher
        query = "simgrep"

        perform_persistent_search(
            query_text=query,
            console=test_console,
            metadata_store=store,
            vector_index=vector_index_val,
            global_config=global_cfg,
            output_mode=OutputMode.show,
            k_results=1,
            min_score=0.1,
        )
        out_one = capsys.readouterr().out
        assert out_one.count("---") == 1

        perform_persistent_search(
            query_text=query,
            console=test_console,
            metadata_store=store,
            vector_index=vector_index_val,
            global_config=global_cfg,
            output_mode=OutputMode.show,
            k_results=2,
            min_score=0.1,
        )
        out_two = capsys.readouterr().out
        assert out_two.count("---") >= 2

    def test_perform_persistent_search_paths_relative_output(
        self,
        populated_persistent_index_for_searcher: Tuple[
            MetadataStore,
            usearch.index.Index,
            SimgrepConfig,
        ],
        test_console: Console,
        capsys: pytest.CaptureFixture,
        persistent_search_test_data_path: Path,
    ) -> None:
        store, vector_index_val, global_cfg = populated_persistent_index_for_searcher
        query = "semantic search"  # matches file2.txt

        perform_persistent_search(
            query_text=query,
            console=test_console,
            metadata_store=store,
            vector_index=vector_index_val,
            global_config=global_cfg,
            output_mode=OutputMode.paths,
            k_results=2,
            display_relative_paths=True,
            base_path_for_relativity=persistent_search_test_data_path,
            min_score=0.25,
        )
        captured = capsys.readouterr()
        assert "file2.txt" in captured.out