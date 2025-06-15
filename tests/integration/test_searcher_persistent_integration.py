from pathlib import Path
from typing import Generator, List, Tuple

import pytest
from rich.console import Console

from simgrep.adapters.sentence_embedder import SentenceEmbedder
from simgrep.adapters.usearch_index import USearchIndex
from simgrep.core.context import SimgrepContext
from simgrep.core.models import OutputMode, SimgrepConfig
from simgrep.indexer import Indexer, IndexerConfig
from simgrep.repository import MetadataStore
from simgrep.services.search_service import SearchService
from simgrep.ui.formatters import format_paths, format_show_basic

pytest.importorskip("transformers")
pytest.importorskip("sentence_transformers")
pytest.importorskip("usearch.index")


@pytest.fixture(scope="session")
def persistent_search_test_data_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Creates a temporary directory with test files, scoped to the session."""
    data_dir = tmp_path_factory.mktemp("persistent_search_data")
    (data_dir / "file1.txt").write_text("simgrep is a tool for semantic search.")
    (data_dir / "file2.txt").write_text("You can use it to find similar text.")
    return data_dir


@pytest.fixture(scope="session")
def default_simgrep_config_for_search_tests(
    tmp_path_factory: pytest.TempPathFactory,
) -> SimgrepConfig:
    """Provides a SimgrepConfig with a temporary data directory for persistent search tests."""
    simgrep_root_config_dir = tmp_path_factory.mktemp("simgrep_config_root_searcher")
    default_proj_data_dir = simgrep_root_config_dir / "default_project"
    default_proj_data_dir.mkdir(parents=True, exist_ok=True)

    cfg = SimgrepConfig(
        default_project_data_dir=default_proj_data_dir,
        default_embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
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
        project_name="searcher_test_project",
        db_path=db_file,
        usearch_index_path=usearch_file,
        embedding_model_name=cfg.default_embedding_model_name,
        chunk_size_tokens=10,
        chunk_overlap_tokens=2,
        file_scan_patterns=["*.txt"],
    )

    indexer_console = Console(quiet=True)
    simgrep_context = SimgrepContext.from_defaults(
        model_name=indexer_config.embedding_model_name,
        chunk_size=indexer_config.chunk_size_tokens,
        chunk_overlap=indexer_config.chunk_overlap_tokens,
    )
    indexer = Indexer(config=indexer_config, context=simgrep_context, console=indexer_console)
    indexer.run_index(target_paths=[persistent_search_test_data_path], wipe_existing=True)

    store = MetadataStore(persistent=True, db_path=db_file)
    vector_index = USearchIndex(ndim=indexer.embedding_ndim)
    vector_index.load(usearch_file)
    embedder = SentenceEmbedder(model_name=cfg.default_embedding_model_name)

    yield store, vector_index, cfg, embedder

    store.close()


class TestSearcherPersistentIntegration:
    @pytest.fixture
    def test_console(self) -> Console:
        """Provides a Rich Console instance for tests, capturing output."""
        return Console(record=True)

    @pytest.mark.parametrize(
        "query, output_mode, expected_strings, min_score",
        [
            pytest.param(
                "semantic search",
                OutputMode.show,
                ["simgrep is a tool for semantic search"],
                0.25,
                id="show_mode",
            ),
            pytest.param(
                "similar text",
                OutputMode.paths,
                ["file2.txt"],
                0.25,
                id="paths_mode",
            ),
            pytest.param(
                "simgrep",
                OutputMode.show,
                ["simgrep is a tool for semantic search", "Score: "],
                0.4,  # High score to filter out the less relevant match
                id="show_mode_high_min_score",
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
                if r.file_path and r.chunk_text:
                    print(format_show_basic(r.file_path, r.chunk_text, r.score))
        elif output_mode == OutputMode.paths:
            print(format_paths([p for r in results if (p := r.file_path)], False, None))

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
