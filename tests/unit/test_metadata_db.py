import pathlib
from typing import Iterator

import duckdb
import pytest

from simgrep.metadata_db import MetadataDBError
from simgrep.metadata_store import MetadataStore
from simgrep.models import ChunkData


@pytest.fixture
def store() -> Iterator[MetadataStore]:
    """Provides a MetadataStore with in-memory DB and ephemeral tables."""
    s = MetadataStore()
    yield s
    s.close()


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
        store = MetadataStore()
        try:
            assert isinstance(store.conn, duckdb.DuckDBPyConnection)
            result = store.conn.execute("SELECT 42;").fetchone()
            assert result is not None
            assert result[0] == 42
        finally:
            store.close()

    def test_setup_ephemeral_tables(self, store: MetadataStore) -> None:
        # check if tables exist
        tables = store.conn.execute("SHOW TABLES;").fetchall()
        table_names = [table[0] for table in tables]
        assert "temp_files" in table_names
        assert "temp_chunks" in table_names

        files_columns = store.conn.execute("DESCRIBE temp_files;").fetchall()
        files_column_names = [col[0] for col in files_columns]
        assert "file_id" in files_column_names
        assert "file_path" in files_column_names

        chunks_columns = store.conn.execute("DESCRIBE temp_chunks;").fetchall()
        chunks_column_names = [col[0] for col in chunks_columns]
        assert "chunk_id" in chunks_column_names
        assert "file_id" in chunks_column_names
        assert "text_content" in chunks_column_names

    def test_batch_insert_files(
        self,
        store: MetadataStore,
        sample_files_metadata: list[tuple[int, pathlib.Path]],
    ) -> None:
        store.batch_insert_files(sample_files_metadata)

        count_result = store.conn.execute("SELECT COUNT(*) FROM temp_files;").fetchone()
        assert count_result is not None
        assert count_result[0] == len(sample_files_metadata)

        file_id_to_check = sample_files_metadata[0][0]
        expected_path = str(sample_files_metadata[0][1].resolve())
        result = store.conn.execute(
            "SELECT file_path FROM temp_files WHERE file_id = ?;", [file_id_to_check]
        ).fetchone()
        assert result is not None
        assert result[0] == expected_path

    def test_batch_insert_files_empty_list(
        self, store: MetadataStore
    ) -> None:
        store.batch_insert_files([])
        count_result = store.conn.execute("SELECT COUNT(*) FROM temp_files;").fetchone()
        assert count_result is not None
        assert count_result[0] == 0

    def test_batch_insert_files_unique_constraint_path(
        self, store: MetadataStore, tmp_path: pathlib.Path
    ) -> None:
        file_path = tmp_path / "unique_test.txt"
        file_path.touch()
        metadata1 = [(0, file_path)]
        store.batch_insert_files(metadata1)

        # attempt to insert the same file path with a different id (should also fail due to unique path)
        # or, more directly for testing unique on file_path:
        metadata2 = [(1, file_path)] # different id, same path
        # temp_files.file_path has a unique constraint
        with pytest.raises(MetadataDBError, match="Failed during batch file insert"):
            store.batch_insert_files(metadata2)

    def test_batch_insert_files_duplicate_file_id(
        self, store: MetadataStore, tmp_path: pathlib.Path
    ) -> None:
        file_path1 = tmp_path / "file_A.txt"
        file_path1.touch()
        file_path2 = tmp_path / "file_B.txt"
        file_path2.touch()

        metadata1 = [(0, file_path1)]
        metadata2 = [(0, file_path2)]  # same file_id, different path

        store.batch_insert_files(metadata1)
        # temp_files.file_id is primary key
        with pytest.raises(MetadataDBError, match="Failed during batch file insert"):
            store.batch_insert_files(metadata2)

    def test_batch_insert_chunks(
        self,
        store: MetadataStore,
        sample_files_metadata: list[tuple[int, pathlib.Path]],
        sample_chunk_data_list: list[ChunkData],
    ) -> None:
        store.batch_insert_files(sample_files_metadata)
        store.batch_insert_chunks(sample_chunk_data_list)

        count_result = store.conn.execute("SELECT COUNT(*) FROM temp_chunks;").fetchone()
        assert count_result is not None
        assert count_result[0] == len(sample_chunk_data_list)

        chunk_to_check = sample_chunk_data_list[0]
        result = store.conn.execute(
            "SELECT text_content, file_id, start_char_offset FROM temp_chunks WHERE chunk_id = ?;",
            [chunk_to_check.usearch_label],
        ).fetchone()
        assert result is not None
        assert result[0] == chunk_to_check.text
        assert result[1] == chunk_to_check.source_file_id
        assert result[2] == chunk_to_check.start_char_offset

    def test_batch_insert_chunks_empty_list(
        self, store: MetadataStore
    ) -> None:
        store.batch_insert_chunks([])
        count_result = store.conn.execute("SELECT COUNT(*) FROM temp_chunks;").fetchone()
        assert count_result is not None
        assert count_result[0] == 0

    def test_batch_insert_chunks_fk_constraint_violation(
        self,
        store: MetadataStore,
        sample_chunk_data_list: list[ChunkData],
    ) -> None:
        with pytest.raises(MetadataDBError, match="Failed during batch chunk insert"):
            store.batch_insert_chunks(sample_chunk_data_list)

    def test_batch_insert_chunks_unique_constraint_chunk_id(
        self,
        store: MetadataStore,
        sample_files_metadata: list[tuple[int, pathlib.Path]],
        sample_chunk_data_list: list[ChunkData],
    ) -> None:
        store.batch_insert_files(sample_files_metadata)

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
            store.batch_insert_chunks(extended_chunk_list)

    def test_retrieve_chunk_for_display_valid_id(
        self,
        store: MetadataStore,
        sample_files_metadata: list[tuple[int, pathlib.Path]],
        sample_chunk_data_list: list[ChunkData],
    ) -> None:
        store.batch_insert_files(sample_files_metadata)
        store.batch_insert_chunks(sample_chunk_data_list)

        chunk_to_retrieve = sample_chunk_data_list[0]
        retrieved = store.retrieve_chunk_for_display(chunk_to_retrieve.usearch_label)

        assert retrieved is not None
        text, path, start_offset, end_offset = retrieved
        assert text == chunk_to_retrieve.text
        assert path == chunk_to_retrieve.source_file_path.resolve()
        assert start_offset == chunk_to_retrieve.start_char_offset
        assert end_offset == chunk_to_retrieve.end_char_offset

    def test_retrieve_chunk_for_display_invalid_id(
        self, store: MetadataStore
    ) -> None:
        retrieved = store.retrieve_chunk_for_display(99999)
        assert retrieved is None

    def test_retrieve_chunk_for_display_chunk_exists_file_missing_in_db(
        self,
        store: MetadataStore,
        sample_files_metadata: list[tuple[int, pathlib.Path]],
        sample_chunk_data_list: list[ChunkData],
    ) -> None:
        # this scenario should ideally be prevented by fk constraints if data is inserted correctly.
        # however, this tests robustness if the db state is somehow inconsistent.

        # insert only the first file
        store.batch_insert_files([sample_files_metadata[0]])

        # insert all chunks. chunks referring to file_id 1 will have a dangling fk if we didn't have constraints.
        # duckdb's default behavior with fks will prevent inserting chunks for file_id 1.
        # so, we'll only insert chunks for file_id 0.
        chunks_for_file0 = [c for c in sample_chunk_data_list if c.source_file_id == 0]
        store.batch_insert_chunks(chunks_for_file0)

        # now, manually remove the file from temp_files after chunks are inserted
        # this simulates an inconsistent state not achievable via normal api if fks are active.
        # to do this, we must temporarily disable fk checks or drop the constraint.
        # for simplicity in a unit test, we'll assume the query in retrieve_chunk_for_display
        # might encounter this if, for example, a file was deleted from temp_files
        # *after* a chunk was inserted (which is bad db management).

        # store the label of a chunk we expect to be deleted
        chunk_to_retrieve_label = chunks_for_file0[0].usearch_label

        # first, delete the chunks that reference file_id = 0
        store.conn.execute("DELETE FROM temp_chunks WHERE file_id = 0;")
        # then, delete the file itself. this should now succeed.
        store.conn.execute("DELETE FROM temp_files WHERE file_id = 0;")

        # try to retrieve a chunk that was linked to the now-deleted file_id 0 and whose record was deleted
        retrieved = store.retrieve_chunk_for_display(chunk_to_retrieve_label)

        # the join should fail to find a match in temp_files (and temp_chunks), so result should be none
        assert (
            retrieved is None
        ), "Should not retrieve chunk if its corresponding file_id is missing from temp_files or chunk itself is deleted"