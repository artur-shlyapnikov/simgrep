import pathlib

import pytest
from rich.console import Console

from simgrep.indexer import Indexer, IndexerConfig, IndexerError
from simgrep.metadata_db import connect_persistent_db
from simgrep.processor import calculate_file_hash
from simgrep.vector_store import load_persistent_index

pytest.importorskip("transformers")
pytest.importorskip("sentence_transformers")
pytest.importorskip("usearch.index")


@pytest.fixture
def temp_project_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Creates a temporary directory for project data (db, index)."""
    project_data_dir = tmp_path / "test_project_data"
    project_data_dir.mkdir(parents=True, exist_ok=True)
    return project_data_dir


@pytest.fixture
def indexer_config(temp_project_dir: pathlib.Path) -> IndexerConfig:
    """Provides a basic IndexerConfig for testing."""
    return IndexerConfig(
        project_name="test_persistent_project",
        db_path=temp_project_dir / "metadata.duckdb",
        usearch_index_path=temp_project_dir / "index.usearch",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size_tokens=128,  # Default from SimgrepConfig
        chunk_overlap_tokens=20,  # Default from SimgrepConfig
        file_scan_patterns=["*.txt", "*.md"],
    )


@pytest.fixture
def test_console() -> Console:
    """Provides a Rich Console for the Indexer."""
    return Console(width=120, quiet=True)  # Quiet to avoid polluting test output


@pytest.fixture
def sample_files_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Creates a directory with sample files for indexing."""
    source_dir = tmp_path / "sample_source_files"
    source_dir.mkdir()

    (source_dir / "file1.txt").write_text("This is the first test file. It contains simple text about simgrep.")
    (source_dir / "file2.md").write_text("# Markdown File\n\nThis is a markdown document with some *emphasis*.")
    (source_dir / "empty.txt").write_text("")
    (source_dir / "noprocess.py").write_text("print('This python file should not be indexed by default patterns')")

    sub_dir = source_dir / "subdir"
    sub_dir.mkdir()
    (sub_dir / "file3.txt").write_text("A file in a subdirectory. More content for simgrep.")

    return source_dir


class TestIndexerPersistent:
    def test_indexer_initialization(self, indexer_config: IndexerConfig, test_console: Console) -> None:
        """Test if the Indexer initializes correctly."""
        try:
            indexer = Indexer(config=indexer_config, console=test_console)
            assert indexer is not None
            assert indexer.config == indexer_config
            assert indexer.embedding_ndim > 0  # Check if ndim was determined
        except IndexerError as e:
            pytest.fail(f"Indexer initialization failed: {e}")

    def test_index_path_new_wipe(
        self,
        indexer_config: IndexerConfig,
        test_console: Console,
        sample_files_dir: pathlib.Path,
    ) -> None:
        """Test indexing a path from scratch with wipe_existing=True."""
        indexer = Indexer(config=indexer_config, console=test_console)

        assert not indexer_config.db_path.exists()
        assert not indexer_config.usearch_index_path.exists()

        try:
            indexer.index_path(target_path=sample_files_dir, wipe_existing=True)
        except IndexerError as e:
            pytest.fail(f"indexer.index_path failed: {e}")

        # Verify database and index files were created
        assert indexer_config.db_path.exists()
        assert indexer_config.usearch_index_path.exists()

        # Verify database content (basic checks)
        db_conn = None
        try:
            db_conn = connect_persistent_db(indexer_config.db_path)

            # Check indexed_files table: file1.txt, file2.md, subdir/file3.txt should be there
            # empty.txt might be skipped if it results in no chunks or is handled as empty.
            # noprocess.py should be skipped due to pattern.
            file_count_result = db_conn.execute("SELECT COUNT(*) FROM indexed_files;").fetchone()
            assert file_count_result is not None
            # Expecting 3 files: file1.txt, file2.md, subdir/file3.txt
            # The exact count depends on how empty files are handled by the indexer logic for DB insertion.
            # If empty files are added to indexed_files, count would be 4.
            # Current Indexer._process_and_index_file skips chunking for empty files but still adds to DB.
            assert file_count_result[0] >= 3

            # Check text_chunks table: should have some chunks
            chunk_count_result = db_conn.execute("SELECT COUNT(*) FROM text_chunks;").fetchone()
            assert chunk_count_result is not None
            assert chunk_count_result[0] > 0  # Expecting some chunks from the non-empty files

            # Verify specific file paths
            expected_file1_path = str((sample_files_dir / "file1.txt").resolve())
            res = db_conn.execute(
                "SELECT COUNT(*) FROM indexed_files WHERE file_path = ?;",
                [expected_file1_path],
            ).fetchone()
            assert res is not None and res[0] == 1

            expected_file3_path = str((sample_files_dir / "subdir" / "file3.txt").resolve())
            res = db_conn.execute(
                "SELECT COUNT(*) FROM indexed_files WHERE file_path = ?;",
                [expected_file3_path],
            ).fetchone()
            assert res is not None and res[0] == 1

            # noprocess.py should not be indexed
            non_indexed_file_path = str((sample_files_dir / "noprocess.py").resolve())
            res = db_conn.execute(
                "SELECT COUNT(*) FROM indexed_files WHERE file_path = ?;",
                [non_indexed_file_path],
            ).fetchone()
            assert res is not None and res[0] == 0

        finally:
            if db_conn:
                db_conn.close()

        # Verify vector index content (basic checks)
        vector_index = load_persistent_index(indexer_config.usearch_index_path)
        assert vector_index is not None
        assert len(vector_index) > 0  # Should have embeddings for the chunks

    def test_index_path_single_file(
        self,
        indexer_config: IndexerConfig,
        test_console: Console,
        sample_files_dir: pathlib.Path,
    ) -> None:
        """Test indexing a single file."""
        indexer = Indexer(config=indexer_config, console=test_console)
        single_file_to_index = sample_files_dir / "file1.txt"

        indexer.index_path(target_path=single_file_to_index, wipe_existing=True)

        assert indexer_config.db_path.exists()
        assert indexer_config.usearch_index_path.exists()

        db_conn = None
        try:
            db_conn = connect_persistent_db(indexer_config.db_path)
            file_count_result = db_conn.execute("SELECT COUNT(*) FROM indexed_files;").fetchone()
            assert file_count_result is not None
            assert file_count_result[0] == 1  # Only one file indexed

            chunk_count_result = db_conn.execute("SELECT COUNT(*) FROM text_chunks;").fetchone()
            assert chunk_count_result is not None
            assert chunk_count_result[0] > 0
        finally:
            if db_conn:
                db_conn.close()

        vector_index = load_persistent_index(indexer_config.usearch_index_path)
        assert vector_index is not None
        assert len(vector_index) > 0

    def test_file_hash_and_mtime_stored(
        self,
        indexer_config: IndexerConfig,
        test_console: Console,
        sample_files_dir: pathlib.Path,
    ) -> None:
        indexer = Indexer(config=indexer_config, console=test_console)
        indexer.index_path(target_path=sample_files_dir, wipe_existing=True)

        file_to_check = sample_files_dir / "file1.txt"

        db_conn = connect_persistent_db(indexer_config.db_path)
        try:
            row = db_conn.execute(
                "SELECT content_hash, last_modified_os FROM indexed_files WHERE file_path = ?;",
                [str(file_to_check.resolve())],
            ).fetchone()
            assert row is not None

            expected_hash = calculate_file_hash(file_to_check)
            db_hash, db_ts = row
            assert db_hash == expected_hash
            assert abs(db_ts.timestamp() - file_to_check.stat().st_mtime) < 1.0
        finally:
            db_conn.close()

    def test_index_empty_directory(
        self,
        indexer_config: IndexerConfig,
        test_console: Console,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test indexing an empty directory."""
        indexer = Indexer(config=indexer_config, console=test_console)
        empty_dir = tmp_path / "empty_test_dir"
        empty_dir.mkdir()

        indexer.index_path(target_path=empty_dir, wipe_existing=True)

        # DB and Index files should still be created (or wiped and recreated empty)
        assert indexer_config.db_path.exists()
        # USearch index might not be saved if it's empty, depending on Indexer logic
        # Current Indexer logic: saves if >0, unlinks if 0. So it should not exist if no files.
        if indexer_config.usearch_index_path.exists():
            idx = load_persistent_index(indexer_config.usearch_index_path)
            assert idx is not None and len(idx) == 0
        else:
            assert not indexer_config.usearch_index_path.exists()

        db_conn = None
        try:
            db_conn = connect_persistent_db(indexer_config.db_path)
            file_count_result = db_conn.execute("SELECT COUNT(*) FROM indexed_files;").fetchone()
            assert file_count_result is not None
            assert file_count_result[0] == 0

            chunk_count_result = db_conn.execute("SELECT COUNT(*) FROM text_chunks;").fetchone()
            assert chunk_count_result is not None
            assert chunk_count_result[0] == 0
        finally:
            if db_conn:
                db_conn.close()

    def test_index_path_non_existent_target(
        self,
        indexer_config: IndexerConfig,
        test_console: Console,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test indexing a non-existent path (should be caught by Typer usually, but test Indexer robustness)."""
        indexer = Indexer(config=indexer_config, console=test_console)
        non_existent_path = tmp_path / "does_not_exist_dir"

        # Indexer.index_path itself doesn't check existence, assumes valid path from CLI.
        # If it proceeds, rglob on non-existent path is fine, returns no files.
        # So, it should behave like an empty directory.
        indexer.index_path(target_path=non_existent_path, wipe_existing=True)

        db_conn = None
        try:
            db_conn = connect_persistent_db(indexer_config.db_path)
            file_count_result = db_conn.execute("SELECT COUNT(*) FROM indexed_files;").fetchone()
            assert file_count_result is not None
            assert file_count_result[0] == 0
        finally:
            if db_conn:
                db_conn.close()

        if indexer_config.usearch_index_path.exists():
            idx = load_persistent_index(indexer_config.usearch_index_path)
            assert idx is not None and len(idx) == 0
        else:
            assert not indexer_config.usearch_index_path.exists()

    # Future tests:
    # - Incremental indexing (add, update, delete files) - requires wipe_existing=False and more logic
    # - Error handling for unreadable files
    # - Different file patterns
    # - Robustness against corrupted existing index/db files (if not wiping)
    # - Indexing very large files or many files (performance, memory)
