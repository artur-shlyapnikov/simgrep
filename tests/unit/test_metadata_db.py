import pathlib
import pytest
import duckdb

from simgrep.models import ChunkData
from simgrep.metadata_db import (
    create_inmemory_db_connection,
    setup_ephemeral_tables,
    batch_insert_files,
    batch_insert_chunks,
    retrieve_chunk_for_display,
)

@pytest.fixture
def db_conn() -> duckdb.DuckDBPyConnection:
    """Fixture to provide an in-memory DuckDB connection with ephemeral tables set up."""
    conn = create_inmemory_db_connection()
    setup_ephemeral_tables(conn)
    yield conn
    conn.close()

@pytest.fixture
def sample_files_metadata(tmp_path: pathlib.Path) -> list[tuple[int, pathlib.Path]]:
    file1 = tmp_path / "file1.txt"
    file1.write_text("Content of file1.")
    file2 = tmp_path / "file2.txt"
    file2.write_text("Content of file2.")
    return [
        (0, file1),
        (1, file2),
    ]

@pytest.fixture
def sample_chunk_data_list(sample_files_metadata: list[tuple[int, pathlib.Path]]) -> list[ChunkData]:
    # file_id 0 corresponds to sample_files_metadata[0] (file1.txt)
    # file_id 1 corresponds to sample_files_metadata[1] (file2.txt)
    return [
        ChunkData(
            text="Chunk 1 from file 1",
            source_file_path=sample_files_metadata[0][1],
            source_file_id=0,
            usearch_label=100,
            start_char_offset=0,
            end_char_offset=19,
            token_count=5,
        ),
        ChunkData(
            text="Chunk 2 from file 1",
            source_file_path=sample_files_metadata[0][1],
            source_file_id=0,
            usearch_label=101,
            start_char_offset=20,
            end_char_offset=39,
            token_count=5,
        ),
        ChunkData(
            text="Chunk 1 from file 2",
            source_file_path=sample_files_metadata[1][1],
            source_file_id=1,
            usearch_label=200,
            start_char_offset=0,
            end_char_offset=19,
            token_count=5,
        ),
    ]

class TestMetadataDB:
    def test_create_inmemory_db_connection(self):
        conn = None
        try:
            conn = create_inmemory_db_connection()
            assert isinstance(conn, duckdb.DuckDBPyConnection)
            assert conn.execute("SELECT 42;").fetchone()[0] == 42
        finally:
            if conn:
                conn.close()

    def test_setup_ephemeral_tables(self, db_conn: duckdb.DuckDBPyConnection):
        # Check if tables exist
        tables = db_conn.execute("SHOW TABLES;").fetchall()
        table_names = [table[0] for table in tables]
        assert "temp_files" in table_names
        assert "temp_chunks" in table_names

        files_columns = db_conn.execute("DESCRIBE temp_files;").fetchall()
        files_column_names = [col[0] for col in files_columns]
        assert "file_id" in files_column_names
        assert "file_path" in files_column_names

        chunks_columns = db_conn.execute("DESCRIBE temp_chunks;").fetchall()
        chunks_column_names = [col[0] for col in chunks_columns]
        assert "chunk_id" in chunks_column_names
        assert "file_id" in chunks_column_names
        assert "text_content" in chunks_column_names

    def test_batch_insert_files(self, db_conn: duckdb.DuckDBPyConnection, sample_files_metadata: list[tuple[int, pathlib.Path]]):
        batch_insert_files(db_conn, sample_files_metadata)
        
        count = db_conn.execute("SELECT COUNT(*) FROM temp_files;").fetchone()[0]
        assert count == len(sample_files_metadata)

        file_id_to_check = sample_files_metadata[0][0]
        expected_path = str(sample_files_metadata[0][1].resolve())
        result = db_conn.execute("SELECT file_path FROM temp_files WHERE file_id = ?;", [file_id_to_check]).fetchone()
        assert result is not None
        assert result[0] == expected_path

    def test_batch_insert_files_empty_list(self, db_conn: duckdb.DuckDBPyConnection):
        batch_insert_files(db_conn, [])
        count = db_conn.execute("SELECT COUNT(*) FROM temp_files;").fetchone()[0]
        assert count == 0
    
    def test_batch_insert_files_unique_constraint_path(self, db_conn: duckdb.DuckDBPyConnection, tmp_path: pathlib.Path):
        file_path = tmp_path / "unique_test.txt"
        file_path.touch()
        metadata1 = [(0, file_path)]
        metadata2 = [(1, file_path)]

        batch_insert_files(db_conn, metadata1)
        with pytest.raises(duckdb.ConstraintException):
            batch_insert_files(db_conn, metadata2)


    def test_batch_insert_chunks(self, db_conn: duckdb.DuckDBPyConnection, sample_files_metadata: list[tuple[int, pathlib.Path]], sample_chunk_data_list: list[ChunkData]):
        # Insert files first to satisfy the foreign key constraint for chunks
        batch_insert_files(db_conn, sample_files_metadata)
        batch_insert_chunks(db_conn, sample_chunk_data_list)

        count = db_conn.execute("SELECT COUNT(*) FROM temp_chunks;").fetchone()[0]
        assert count == len(sample_chunk_data_list)

        chunk_to_check = sample_chunk_data_list[0]
        result = db_conn.execute("SELECT text_content, file_id, start_char_offset FROM temp_chunks WHERE chunk_id = ?;", [chunk_to_check.usearch_label]).fetchone()
        assert result is not None
        assert result[0] == chunk_to_check.text
        assert result[1] == chunk_to_check.source_file_id
        assert result[2] == chunk_to_check.start_char_offset

    def test_batch_insert_chunks_empty_list(self, db_conn: duckdb.DuckDBPyConnection):
        batch_insert_chunks(db_conn, [])
        count = db_conn.execute("SELECT COUNT(*) FROM temp_chunks;").fetchone()[0]
        assert count == 0

    def test_batch_insert_chunks_fk_constraint(self, db_conn: duckdb.DuckDBPyConnection, sample_chunk_data_list: list[ChunkData]):
        with pytest.raises(duckdb.ConstraintException):
            batch_insert_chunks(db_conn, sample_chunk_data_list)

    def test_retrieve_chunk_for_display_valid_id(self, db_conn: duckdb.DuckDBPyConnection, sample_files_metadata: list[tuple[int, pathlib.Path]], sample_chunk_data_list: list[ChunkData]):
        batch_insert_files(db_conn, sample_files_metadata)
        batch_insert_chunks(db_conn, sample_chunk_data_list)

        chunk_to_retrieve = sample_chunk_data_list[0]
        retrieved = retrieve_chunk_for_display(db_conn, chunk_to_retrieve.usearch_label)

        assert retrieved is not None
        text, path, start_offset, end_offset = retrieved
        assert text == chunk_to_retrieve.text
        assert path == chunk_to_retrieve.source_file_path.resolve()
        assert start_offset == chunk_to_retrieve.start_char_offset
        assert end_offset == chunk_to_retrieve.end_char_offset

    def test_retrieve_chunk_for_display_invalid_id(self, db_conn: duckdb.DuckDBPyConnection):
        retrieved = retrieve_chunk_for_display(db_conn, 99999) # Non-existent ID
        assert retrieved is None