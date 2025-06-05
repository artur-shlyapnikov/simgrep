import pathlib
from typing import Iterator

import duckdb
import pytest

from simgrep.metadata_db import (
    MetadataDBError,
    batch_insert_chunks,
    batch_insert_files,
    create_inmemory_db_connection,
    retrieve_chunk_for_display,
    retrieve_chunks_for_display_bulk,
    setup_ephemeral_tables,
)
from simgrep.models import ChunkData


@pytest.fixture
def db_conn() -> Iterator[duckdb.DuckDBPyConnection]:
    """fixture to provide an in-memory DuckDB connection with ephemeral tables set up."""
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
def sample_chunk_data_list(
    sample_files_metadata: list[tuple[int, pathlib.Path]],
) -> list[ChunkData]:
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
    def test_create_inmemory_db_connection(self) -> None:
        conn = None
        try:
            conn = create_inmemory_db_connection()
            assert isinstance(conn, duckdb.DuckDBPyConnection)
            result = conn.execute("SELECT 42;").fetchone()
            assert result is not None
            assert result[0] == 42
        finally:
            if conn:
                conn.close()

    def test_setup_ephemeral_tables(self, db_conn: duckdb.DuckDBPyConnection) -> None:
        # check if tables exist
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

    def test_batch_insert_files(
        self,
        db_conn: duckdb.DuckDBPyConnection,
        sample_files_metadata: list[tuple[int, pathlib.Path]],
    ) -> None:
        batch_insert_files(db_conn, sample_files_metadata)

        count_result = db_conn.execute("SELECT COUNT(*) FROM temp_files;").fetchone()
        assert count_result is not None
        assert count_result[0] == len(sample_files_metadata)

        file_id_to_check = sample_files_metadata[0][0]
        expected_path = str(sample_files_metadata[0][1].resolve())
        result = db_conn.execute("SELECT file_path FROM temp_files WHERE file_id = ?;", [file_id_to_check]).fetchone()
        assert result is not None
        assert result[0] == expected_path

    def test_batch_insert_files_empty_list(self, db_conn: duckdb.DuckDBPyConnection) -> None:
        batch_insert_files(db_conn, [])
        count_result = db_conn.execute("SELECT COUNT(*) FROM temp_files;").fetchone()
        assert count_result is not None
        assert count_result[0] == 0

    def test_batch_insert_files_unique_constraint_path(self, db_conn: duckdb.DuckDBPyConnection, tmp_path: pathlib.Path) -> None:
        file_path = tmp_path / "unique_test.txt"
        file_path.touch()
        metadata1 = [(0, file_path)]
        batch_insert_files(db_conn, metadata1)

        # attempt to insert the same file path with a different id (should also fail due to unique path)
        # or, more directly for testing unique on file_path:
        metadata2 = [(1, file_path)]  # different id, same path
        # temp_files.file_path has a unique constraint
        with pytest.raises(MetadataDBError, match="Failed during batch file insert"):
            batch_insert_files(db_conn, metadata2)

    def test_batch_insert_files_duplicate_file_id(self, db_conn: duckdb.DuckDBPyConnection, tmp_path: pathlib.Path) -> None:
        file_path1 = tmp_path / "file_A.txt"
        file_path1.touch()
        file_path2 = tmp_path / "file_B.txt"
        file_path2.touch()

        metadata1 = [(0, file_path1)]
        metadata2 = [(0, file_path2)]  # same file_id, different path

        batch_insert_files(db_conn, metadata1)
        # temp_files.file_id is primary key
        with pytest.raises(MetadataDBError, match="Failed during batch file insert"):
            batch_insert_files(db_conn, metadata2)

    def test_batch_insert_chunks(
        self,
        db_conn: duckdb.DuckDBPyConnection,
        sample_files_metadata: list[tuple[int, pathlib.Path]],
        sample_chunk_data_list: list[ChunkData],
    ) -> None:
        # insert files first to satisfy the foreign key constraint for chunks
        batch_insert_files(db_conn, sample_files_metadata)
        batch_insert_chunks(db_conn, sample_chunk_data_list)

        count_result = db_conn.execute("SELECT COUNT(*) FROM temp_chunks;").fetchone()
        assert count_result is not None
        assert count_result[0] == len(sample_chunk_data_list)

        chunk_to_check = sample_chunk_data_list[0]
        result = db_conn.execute(
            "SELECT text_content, file_id, start_char_offset FROM temp_chunks WHERE chunk_id = ?;",
            [chunk_to_check.usearch_label],
        ).fetchone()
        assert result is not None
        assert result[0] == chunk_to_check.text
        assert result[1] == chunk_to_check.source_file_id
        assert result[2] == chunk_to_check.start_char_offset

    def test_batch_insert_chunks_empty_list(self, db_conn: duckdb.DuckDBPyConnection) -> None:
        batch_insert_chunks(db_conn, [])
        count_result = db_conn.execute("SELECT COUNT(*) FROM temp_chunks;").fetchone()
        assert count_result is not None
        assert count_result[0] == 0

    def test_batch_insert_chunks_fk_constraint_violation(
        self,
        db_conn: duckdb.DuckDBPyConnection,
        sample_chunk_data_list: list[ChunkData],
    ) -> None:
        # files are not inserted, so fk constraint on temp_chunks.file_id will fail
        with pytest.raises(MetadataDBError, match="Failed during batch chunk insert"):
            batch_insert_chunks(db_conn, sample_chunk_data_list)

    def test_batch_insert_chunks_unique_constraint_chunk_id(
        self,
        db_conn: duckdb.DuckDBPyConnection,
        sample_files_metadata: list[tuple[int, pathlib.Path]],
        sample_chunk_data_list: list[ChunkData],
    ) -> None:
        batch_insert_files(db_conn, sample_files_metadata)

        # create a duplicate chunk_id (usearch_label)
        chunk_with_duplicate_id = ChunkData(
            text="Chunk 1 from file 1",
            source_file_path=sample_chunk_data_list[0].source_file_path,
            source_file_id=sample_chunk_data_list[0].source_file_id,
            usearch_label=sample_chunk_data_list[0].usearch_label,  # duplicate id
            start_char_offset=100,
            end_char_offset=110,
            token_count=3,
        )
        extended_chunk_list = sample_chunk_data_list + [chunk_with_duplicate_id]

        # temp_chunks.chunk_id is primary key
        with pytest.raises(MetadataDBError, match="Failed during batch chunk insert"):
            batch_insert_chunks(db_conn, extended_chunk_list)

    def test_retrieve_chunk_for_display_valid_id(
        self,
        db_conn: duckdb.DuckDBPyConnection,
        sample_files_metadata: list[tuple[int, pathlib.Path]],
        sample_chunk_data_list: list[ChunkData],
    ) -> None:
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

    def test_retrieve_chunk_for_display_invalid_id(self, db_conn: duckdb.DuckDBPyConnection) -> None:
        retrieved = retrieve_chunk_for_display(db_conn, 99999)  # non-existent id
        assert retrieved is None

    def test_retrieve_chunk_for_display_chunk_exists_file_missing_in_db(
        self,
        db_conn: duckdb.DuckDBPyConnection,
        sample_files_metadata: list[tuple[int, pathlib.Path]],
        sample_chunk_data_list: list[ChunkData],
    ) -> None:
        # this scenario should ideally be prevented by fk constraints if data is inserted correctly.
        # however, this tests robustness if the db state is somehow inconsistent.

        # insert only the first file
        batch_insert_files(db_conn, [sample_files_metadata[0]])

        # insert all chunks. chunks referring to file_id 1 will have a dangling fk if we didn't have constraints.
        # duckdb's default behavior with fks will prevent inserting chunks for file_id 1.
        # so, we'll only insert chunks for file_id 0.
        chunks_for_file0 = [c for c in sample_chunk_data_list if c.source_file_id == 0]
        batch_insert_chunks(db_conn, chunks_for_file0)

        # now, manually remove the file from temp_files after chunks are inserted
        # this simulates an inconsistent state not achievable via normal api if fks are active.
        # to do this, we must temporarily disable fk checks or drop the constraint.
        # for simplicity in a unit test, we'll assume the query in retrieve_chunk_for_display
        # might encounter this if, for example, a file was deleted from temp_files
        # *after* a chunk was inserted (which is bad db management).

        # store the label of a chunk we expect to be deleted
        chunk_to_retrieve_label = chunks_for_file0[0].usearch_label

        # first, delete the chunks that reference file_id = 0
        db_conn.execute("DELETE FROM temp_chunks WHERE file_id = 0;")
        # then, delete the file itself. this should now succeed.
        db_conn.execute("DELETE FROM temp_files WHERE file_id = 0;")

        # try to retrieve a chunk that was linked to the now-deleted file_id 0 and whose record was deleted
        retrieved = retrieve_chunk_for_display(db_conn, chunk_to_retrieve_label)

        # the join should fail to find a match in temp_files (and temp_chunks), so result should be none
        assert retrieved is None, (
            "Should not retrieve chunk if its corresponding file_id is missing from temp_files or chunk itself is deleted"
        )

    def test_retrieve_chunks_for_display_bulk(
        self,
        db_conn: duckdb.DuckDBPyConnection,
        sample_files_metadata: list[tuple[int, pathlib.Path]],
        sample_chunk_data_list: list[ChunkData],
    ) -> None:
        batch_insert_files(db_conn, sample_files_metadata)
        batch_insert_chunks(db_conn, sample_chunk_data_list)

        ids = [c.usearch_label for c in sample_chunk_data_list]
        results = retrieve_chunks_for_display_bulk(db_conn, ids)

        assert len(results) == len(ids)
        for chunk in sample_chunk_data_list:
            assert chunk.usearch_label in results
            text, path, start, end = results[chunk.usearch_label]
            assert text == chunk.text
            assert path == chunk.source_file_path.resolve()
            assert start == chunk.start_char_offset
            assert end == chunk.end_char_offset

    def test_retrieve_chunks_for_display_bulk_with_missing(
        self,
        db_conn: duckdb.DuckDBPyConnection,
        sample_files_metadata: list[tuple[int, pathlib.Path]],
        sample_chunk_data_list: list[ChunkData],
    ) -> None:
        batch_insert_files(db_conn, sample_files_metadata)
        batch_insert_chunks(db_conn, sample_chunk_data_list)

        ids = [sample_chunk_data_list[0].usearch_label, 99999]
        results = retrieve_chunks_for_display_bulk(db_conn, ids)

        assert sample_chunk_data_list[0].usearch_label in results
        assert 99999 not in results
