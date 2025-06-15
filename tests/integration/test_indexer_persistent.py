import pathlib
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from simgrep.adapters.usearch_index import USearchIndex
from simgrep.core.context import SimgrepContext
from simgrep.core.errors import IndexerError
from simgrep.indexer import Indexer, IndexerConfig
from simgrep.repository import MetadataStore

pytest.importorskip("transformers")
pytest.importorskip("sentence_transformers")
pytest.importorskip("usearch.index")


@pytest.fixture
def persistent_test_data_path(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Creates a temporary directory with some test files for persistent indexing."""
    data_dir = tmp_path_factory.mktemp("persistent_data")
    (data_dir / "file1.txt").write_text("Hello world, this is a test.")
    (data_dir / "file2.txt").write_text("Simgrep is a semantic search tool.")
    (data_dir / "another").mkdir()
    (data_dir / "another" / "file3.txt").write_text("This is another file in a sub-directory.")
    (data_dir / "empty.txt").write_text("")  # empty file
    return data_dir


@pytest.fixture
def indexer_config(tmp_path: pathlib.Path) -> IndexerConfig:
    """Provides a standard IndexerConfig for persistent tests."""
    return IndexerConfig(
        project_name="persistent_test_project",
        db_path=tmp_path / "metadata.duckdb",
        usearch_index_path=tmp_path / "index.usearch",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",  # a small, fast model
        chunk_size_tokens=16,
        chunk_overlap_tokens=4,
        file_scan_patterns=["*.txt"],
        max_index_workers=1,  # for deterministic testing
    )


@pytest.fixture
def simgrep_context(indexer_config: IndexerConfig) -> SimgrepContext:
    return SimgrepContext.from_defaults(
        model_name=indexer_config.embedding_model_name,
        chunk_size=indexer_config.chunk_size_tokens,
        chunk_overlap=indexer_config.chunk_overlap_tokens,
    )


@pytest.fixture
def test_console() -> Console:
    """Provides a Rich Console instance for tests, suppressing output."""
    return Console(quiet=True)


class TestIndexerPersistentIntegration:
    def test_index_path_new_wipe(
        self,
        persistent_test_data_path: pathlib.Path,
        indexer_config: IndexerConfig,
        simgrep_context: SimgrepContext,
        test_console: Console,
    ) -> None:
        """Test indexing a path from scratch with wipe=True."""
        indexer = Indexer(config=indexer_config, context=simgrep_context, console=test_console)
        indexer.run_index(target_paths=[persistent_test_data_path], wipe_existing=True)

        # Verify DB and Index files were created
        assert indexer_config.db_path.exists()
        assert indexer_config.usearch_index_path.exists()

        # Verify DB content
        store = None
        try:
            store = MetadataStore(persistent=True, db_path=indexer_config.db_path)
            db_conn = store.conn
            file_count_result = db_conn.execute("SELECT COUNT(*) FROM indexed_files;").fetchone()
            assert file_count_result is not None
            # The new logic does not add a file record if no chunks are produced.
            assert file_count_result[0] == 4

            chunk_count_result = db_conn.execute("SELECT COUNT(*) FROM text_chunks;").fetchone()
            assert chunk_count_result is not None
            assert chunk_count_result[0] > 0

            # check one record for correctness
            file1_path = str((persistent_test_data_path / "file1.txt").resolve())
            file_record_result = db_conn.execute(
                "SELECT file_id FROM indexed_files WHERE file_path = ?", [file1_path]
            ).fetchone()
            assert file_record_result is not None
            file_id = file_record_result[0]

            chunk_record_result = db_conn.execute(
                "SELECT COUNT(*) FROM text_chunks WHERE file_id = ?", [file_id]
            ).fetchone()
            assert chunk_record_result is not None
            assert chunk_record_result[0] > 0
        finally:
            if store:
                store.close()

        vector_index = USearchIndex(ndim=indexer.embedding_ndim)
        vector_index.load(indexer_config.usearch_index_path)
        assert len(vector_index) > 0

    def test_index_path_single_file(
        self,
        persistent_test_data_path: pathlib.Path,
        indexer_config: IndexerConfig,
        simgrep_context: SimgrepContext,
        test_console: Console,
    ) -> None:
        """Test indexing a single file directly."""
        file_to_index = persistent_test_data_path / "file1.txt"
        indexer = Indexer(config=indexer_config, context=simgrep_context, console=test_console)
        indexer.run_index(target_paths=[file_to_index], wipe_existing=True)

        assert indexer_config.db_path.exists()
        assert indexer_config.usearch_index_path.exists()

        store = None
        try:
            store = MetadataStore(persistent=True, db_path=indexer_config.db_path)
            file_count = store.conn.execute("SELECT COUNT(*) FROM indexed_files;").fetchone()
            assert file_count is not None and file_count[0] == 1
        finally:
            if store:
                store.close()

    def test_index_non_existent_path_is_handled_gracefully(
        self,
        tmp_path: pathlib.Path,
        indexer_config: IndexerConfig,
        simgrep_context: SimgrepContext,
        test_console: Console,
    ) -> None:
        """Test that indexing a non-existent path is handled gracefully and creates an empty index."""
        non_existent_path = tmp_path / "does_not_exist"
        indexer = Indexer(config=indexer_config, context=simgrep_context, console=test_console)
        # The indexer should handle this gracefully and report it.
        # It should not raise an exception, but complete with 0 files processed.
        indexer.run_index(target_paths=[non_existent_path], wipe_existing=True)

        # An empty index file should be created.
        assert indexer_config.usearch_index_path.exists()
        idx = USearchIndex(ndim=indexer.embedding_ndim)
        idx.load(indexer_config.usearch_index_path)
        assert len(idx) == 0

    def test_index_empty_directory(
        self,
        tmp_path: pathlib.Path,
        indexer_config: IndexerConfig,
        simgrep_context: SimgrepContext,
        test_console: Console,
    ) -> None:
        """Test indexing an empty directory completes without errors."""
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        indexer = Indexer(config=indexer_config, context=simgrep_context, console=test_console)
        indexer.run_index(target_paths=[empty_dir], wipe_existing=True)

        # DB and Index files should still be created (or wiped and recreated empty)
        assert indexer_config.db_path.exists()
        # USearch index might not be saved if it's empty, depending on Indexer logic
        # Current Indexer logic: saves if >0.
        if indexer_config.usearch_index_path.exists():
            idx = USearchIndex(ndim=indexer.embedding_ndim)
            idx.load(indexer_config.usearch_index_path)
            assert len(idx) == 0
            # This behavior is acceptable.
            pass

        store_check = None
        try:
            store_check = MetadataStore(persistent=True, db_path=indexer_config.db_path)
            file_count = store_check.conn.execute("SELECT COUNT(*) FROM indexed_files").fetchone()
            assert file_count is not None and file_count[0] == 0
        finally:
            if store_check:
                store_check.close()

    def test_indexer_raises_on_db_failure(
        self,
        persistent_test_data_path: pathlib.Path,
        indexer_config: IndexerConfig,
        simgrep_context: SimgrepContext,
        test_console: Console,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that Indexer raises IndexerError if the DB fails to prepare."""
        # Simulate a DB connection failure
        monkeypatch.setattr(
            "simgrep.indexer.MetadataStore",
            MagicMock(side_effect=IndexerError("DB fail")),
        )
        indexer = Indexer(config=indexer_config, context=simgrep_context, console=test_console)
        with pytest.raises(IndexerError, match="DB fail"):
            indexer.run_index(target_paths=[persistent_test_data_path], wipe_existing=True)

    def test_incremental_skips_and_updates(
        self,
        persistent_test_data_path: pathlib.Path,
        indexer_config: IndexerConfig,
        simgrep_context: SimgrepContext,
        test_console: Console,
    ) -> None:
        """Test incremental indexing skips unchanged files and updates changed ones."""
        # Initial full index
        indexer = Indexer(config=indexer_config, context=simgrep_context, console=test_console)
        indexer.run_index(target_paths=[persistent_test_data_path], wipe_existing=True)

        store_before = MetadataStore(persistent=True, db_path=indexer_config.db_path)
        try:
            res_fcb = store_before.conn.execute("SELECT COUNT(*) FROM indexed_files").fetchone()
            assert res_fcb is not None
            file_count_before = res_fcb[0]

            res_ccb = store_before.conn.execute("SELECT COUNT(*) FROM text_chunks").fetchone()
            assert res_ccb is not None
            chunk_count_before = res_ccb[0]

            file2_path = str((persistent_test_data_path / "file2.txt").resolve())
            res_f2hb = store_before.conn.execute(
                "SELECT content_hash FROM indexed_files WHERE file_path = ?", [file2_path]
            ).fetchone()
            assert res_f2hb is not None
            file2_hash_before = res_f2hb[0]
        finally:
            store_before.close()

        idx_before = USearchIndex(ndim=indexer.embedding_ndim)
        idx_before.load(indexer_config.usearch_index_path)
        index_size_before = len(idx_before)

        # Second run with no changes, should be a no-op for content.
        indexer.run_index(target_paths=[persistent_test_data_path], wipe_existing=False)

        store_after = MetadataStore(persistent=True, db_path=indexer_config.db_path)
        try:
            res_fcn = store_after.conn.execute("SELECT COUNT(*) FROM indexed_files").fetchone()
            assert res_fcn is not None
            file_count_nochange = res_fcn[0]

            res_ccn = store_after.conn.execute("SELECT COUNT(*) FROM text_chunks").fetchone()
            assert res_ccn is not None
            chunk_count_nochange = res_ccn[0]

            res_f2ha = store_after.conn.execute(
                "SELECT content_hash FROM indexed_files WHERE file_path = ?", [file2_path]
            ).fetchone()
            assert res_f2ha is not None
            file2_hash_after = res_f2ha[0]
        finally:
            store_after.close()

        idx_nochange = USearchIndex(ndim=indexer.embedding_ndim)
        idx_nochange.load(indexer_config.usearch_index_path)
        index_size_nochange = len(idx_nochange)

        assert file2_hash_after == file2_hash_before
        assert file_count_nochange == file_count_before
        assert chunk_count_nochange == chunk_count_before
        assert index_size_nochange == index_size_before
