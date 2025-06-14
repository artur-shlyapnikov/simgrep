from pathlib import Path
from typing import Generator, List, Tuple

import pytest
import usearch.index
from rich.console import Console

from simgrep.adapters.sentence_embedder import SentenceEmbedder
from simgrep.adapters.usearch_index import USearchIndex
from simgrep.core.models import OutputMode, SimgrepConfig
from simgrep.indexer import Indexer, IndexerConfig
from simgrep.repository import MetadataStore
from simgrep.services.search_service import SearchService
from simgrep.ui.formatters import format_paths, format_show_basic

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

    file1_content = "This file discusses advanced information retrieval techniques. A very specific keyword is xyz_super_unique."
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
    default_proj_data_dir.mkdir(parents=True, exist_ok=True)

    cfg = SimgrepConfig(default_project_data_dir=default_proj_data_dir)
    return cfg


@pytest.fixture(scope="session")
def populated_persistent_index_for_searcher(
    persistent_search_test_data_path: Path,
    default_simgrep_config_for_search_tests: SimgrepConfig,
) -> Generator[Tuple[MetadataStore, USearchIndex, SimgrepConfig, SentenceEmbedder], None, None]:
    """
    Indexes dummy data and provides the DB connection, USearch index, and config for tests.
    This is session-scoped for efficiency as indexing can be slow.
    """
    cfg = default_simgrep_config_for_search_tests
    db_file = cfg.default_project_data_dir / "test_searcher_fixture_metadata.duckdb"
    usearch_file = cfg.default_project_data_dir / "test_searcher_fixture_index.usearch"

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

    indexer_console = Console(quiet=True)
    indexer = Indexer(config=indexer_config, console=indexer_console)
    indexer.run_index(target_paths=[persistent_search_test_data_path], wipe_existing=True)

    store = MetadataStore(persistent=True, db_path=db_file)
    vector_index = USearchIndex(ndim=indexer.embedding_ndim)
    vector_index.load(usearch_file)
    embedder = SentenceEmbedder(model_name=cfg.default_embedding_model_name)

    yield store, vector_index, cfg, embedder

    store.close()


class TestSearcherPersistentIntegration:
    @pytest.mark.parametrize(
        "query, output_mode, expected_strings, min_score",
        [
            pytest.param(
                "xyz_super_unique",
                OutputMode.show,
                ["File:", "Score:", "Chunk:", "file1.txt"],
                0.25,
                id="show_mode_with_results",
            ),
            pytest.param(
                "semantic search",
                OutputMode.paths,
                ["file2.txt"],
                0.25,
                id="paths_mode_with_results",
            ),
        ],
    )
    def test_search_service_with_results(
        self,
        populated_persistent_index_for_searcher: Tuple[MetadataStore, USearchIndex, SimgrepConfig, SentenceEmbedder],
        test_console: Console,
        capsys: pytest.CaptureFixture,
        query: str,
        output_mode: OutputMode,
        expected_strings: List[str],
        min_score: float,
    ) -> None:
        """Tests that SearchService returns expected results."""
        store, vector_index, cfg, embedder = populated_persistent_index_for_searcher
        service = SearchService(store=store, embedder=embedder, index=vector_index)

        results = service.search(
            query=query,
            k=2,
            min_score=min_score,
            file_filter=None,
            keyword_filter=None,
        )

        assert results
        if output_mode == OutputMode.show:
            for r in results:
                test_console.print(format_show_basic(r["file_path"], r["chunk_text"], r["score"]))
        elif output_mode == OutputMode.paths:
            test_console.print(format_paths([r["file_path"] for r in results], False, None))

        captured = capsys.readouterr()
        output_str = captured.out.replace("\n", "")
        for expected_str in expected_strings:
            assert expected_str in output_str

    def test_search_service_no_results(
        self,
        populated_persistent_index_for_searcher: Tuple[MetadataStore, USearchIndex, SimgrepConfig, SentenceEmbedder],
    ) -> None:
        """Tests that SearchService handles no-result scenarios gracefully."""
        store, vector_index, cfg, embedder = populated_persistent_index_for_searcher
        service = SearchService(store=store, embedder=embedder, index=vector_index)
        query = "zzxxyy_non_existent_term_qwerty_12345"

        results = service.search(
            query=query,
            k=2,
            min_score=0.95,
            file_filter=None,
            keyword_filter=None,
        )
        assert not results
