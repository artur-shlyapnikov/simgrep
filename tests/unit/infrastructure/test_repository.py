import pathlib
from typing import Any, Dict, Iterator, List, Tuple

import duckdb
import pytest

pytest.mark.external

# Ensure consistent exception type
from simgrep.core.errors import MetadataDBError
from simgrep.core.models import ChunkData
from simgrep.repository import MetadataStore


@pytest.fixture
def ephemeral_store() -> Iterator[MetadataStore]:
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
            text="Chunk 2 from file 1 with KEYWORD",
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


class TestMetadataStoreInit:
    def test_init_ephemeral(self):
        store = MetadataStore(persistent=False)
        assert store.conn.execute("SELECT current_database();").fetchone()[0] == "memory"
        store.close()

    def test_init_persistent_no_path_raises(self):
        with pytest.raises(ValueError, match="db_path must be provided for persistent MetadataStore"):
            MetadataStore(persistent=True, db_path=None)

    def test_init_ephemeral_with_path_is_ignored(self, tmp_path: pathlib.Path):
        store = MetadataStore(persistent=False, db_path=tmp_path / "db.duckdb")
        assert store.conn.execute("SELECT current_database();").fetchone()[0] == "memory"
        assert not (tmp_path / "db.duckdb").exists()
        store.close()


class TestMetadataStoreEphemeral:
    def test_batch_insert_files(self, ephemeral_store: MetadataStore, sample_files_metadata: list[tuple[int, pathlib.Path]]) -> None:
        ephemeral_store.batch_insert_files(sample_files_metadata)
        count_result = ephemeral_store.conn.execute("SELECT COUNT(*) FROM temp_files;").fetchone()
        assert count_result[0] == len(sample_files_metadata)

    def test_batch_insert_chunks(
        self, ephemeral_store: MetadataStore, sample_files_metadata: list[tuple[int, pathlib.Path]], sample_chunk_data_list: list[ChunkData]
    ) -> None:
        ephemeral_store.batch_insert_files(sample_files_metadata)
        ephemeral_store.batch_insert_chunks(sample_chunk_data_list)
        count_result = ephemeral_store.conn.execute("SELECT COUNT(*) FROM temp_chunks;").fetchone()
        assert count_result[0] == len(sample_chunk_data_list)

    def test_retrieve_chunk_for_display_valid_id(
        self, ephemeral_store: MetadataStore, sample_files_metadata: list[tuple[int, pathlib.Path]], sample_chunk_data_list: list[ChunkData]
    ) -> None:
        ephemeral_store.batch_insert_files(sample_files_metadata)
        ephemeral_store.batch_insert_chunks(sample_chunk_data_list)
        chunk_to_retrieve = sample_chunk_data_list[0]
        retrieved = ephemeral_store.retrieve_chunk_for_display(chunk_to_retrieve.usearch_label)
        assert retrieved is not None
        text, path, start, end = retrieved
        assert text == chunk_to_retrieve.text
        assert path == chunk_to_retrieve.source_file_path.resolve()


@pytest.fixture
def persistent_store(tmp_path: pathlib.Path) -> Iterator[MetadataStore]:
    db_path = tmp_path / "persistent_test.duckdb"
    store = MetadataStore(persistent=True, db_path=db_path)
    yield store
    store.close()


@pytest.fixture
def populated_persistent_store(persistent_store: MetadataStore, tmp_path: pathlib.Path) -> MetadataStore:
    # file 1
    file1_path = tmp_path / "file1.txt"
    file1_path.write_text("content")
    file1_id = persistent_store.insert_indexed_file_record(str(file1_path), "h1", 7, 1.0)
    # file 2
    file2_path = tmp_path / "file2.py"
    file2_path.write_text("python content")
    file2_id = persistent_store.insert_indexed_file_record(str(file2_path), "h2", 12, 1.0)
    # file 3 (no chunks)
    file3_path = tmp_path / "file3.txt"
    file3_path.write_text("empty")
    persistent_store.insert_indexed_file_record(str(file3_path), "h3", 5, 1.0)

    chunks = [
        {"file_id": file1_id, "usearch_label": 10, "chunk_text": "content", "start_char_offset": 0, "end_char_offset": 7, "token_count": 1},
        {"file_id": file2_id, "usearch_label": 20, "chunk_text": "python", "start_char_offset": 0, "end_char_offset": 6, "token_count": 1},
        {"file_id": file2_id, "usearch_label": 21, "chunk_text": "KEYWORD", "start_char_offset": 7, "end_char_offset": 12, "token_count": 1},
    ]
    persistent_store.batch_insert_text_chunks(chunks)
    persistent_store.set_max_usearch_label(21)
    return persistent_store


class TestMetadataStorePersistent:
    def test_retrieve_chunk_details_persistent(self, populated_persistent_store: MetadataStore, tmp_path: pathlib.Path) -> None:
        retrieved = populated_persistent_store.retrieve_chunk_details_persistent(usearch_label=10)
        assert retrieved is not None
        text, path, start, end = retrieved
        assert text == "content"
        assert path == (tmp_path / "file1.txt").resolve()

    def test_retrieve_chunk_details_persistent_non_existent(self, populated_persistent_store: MetadataStore) -> None:
        assert populated_persistent_store.retrieve_chunk_details_persistent(999) is None

    def test_retrieve_filtered_chunk_details_no_filters(self, populated_persistent_store: MetadataStore) -> None:
        results = populated_persistent_store.retrieve_filtered_chunk_details(usearch_labels=[10, 20])
        assert len(results) == 2
        assert {r["usearch_label"] for r in results} == {10, 20}

    def test_retrieve_filtered_chunk_details_empty_labels(self, populated_persistent_store: MetadataStore) -> None:
        assert populated_persistent_store.retrieve_filtered_chunk_details(usearch_labels=[]) == []

    def test_retrieve_filtered_chunk_details_file_filter(self, populated_persistent_store: MetadataStore) -> None:
        results = populated_persistent_store.retrieve_filtered_chunk_details(usearch_labels=[10, 20, 21], file_filter=["*.py"])
        assert len(results) == 2
        assert {r["usearch_label"] for r in results} == {20, 21}

    def test_retrieve_filtered_chunk_details_keyword_filter(self, populated_persistent_store: MetadataStore) -> None:
        results = populated_persistent_store.retrieve_filtered_chunk_details(usearch_labels=[10, 20, 21], keyword_filter="keyword")
        assert len(results) == 1
        assert results[0]["usearch_label"] == 21

    def test_retrieve_filtered_chunk_details_combined_filters(self, populated_persistent_store: MetadataStore) -> None:
        results = populated_persistent_store.retrieve_filtered_chunk_details(usearch_labels=[10, 20, 21], file_filter=["*.py"], keyword_filter="python")
        assert len(results) == 1
        assert results[0]["usearch_label"] == 20

    def test_clear_persistent_project_data(self, populated_persistent_store: MetadataStore) -> None:
        populated_persistent_store.clear_persistent_project_data()
        files, chunks = populated_persistent_store.get_index_counts()
        assert files == 0
        assert chunks == 0

    def test_delete_file_records(self, populated_persistent_store: MetadataStore, tmp_path: pathlib.Path) -> None:
        file1_id = populated_persistent_store.conn.execute("SELECT file_id FROM indexed_files WHERE file_path = ?", [str(tmp_path / "file1.txt")]).fetchone()[0]
        deleted_labels = populated_persistent_store.delete_file_records(file1_id)
        assert deleted_labels == [10]
        files, chunks = populated_persistent_store.get_index_counts()
        assert files == 2
        assert chunks == 2

    def test_delete_file_records_no_chunks(self, populated_persistent_store: MetadataStore, tmp_path: pathlib.Path) -> None:
        file3_id = populated_persistent_store.conn.execute("SELECT file_id FROM indexed_files WHERE file_path = ?", [str(tmp_path / "file3.txt")]).fetchone()[0]
        deleted_labels = populated_persistent_store.delete_file_records(file3_id)
        assert deleted_labels == []
        files, chunks = populated_persistent_store.get_index_counts()
        assert files == 2
        assert chunks == 3

    def test_get_and_set_max_usearch_label(self, persistent_store: MetadataStore) -> None:
        assert persistent_store.get_max_usearch_label() is None
        persistent_store.set_max_usearch_label(42)
        assert persistent_store.get_max_usearch_label() == 42
        persistent_store.set_max_usearch_label(100)
        assert persistent_store.get_max_usearch_label() == 100
        # Test ON CONFLICT...DO UPDATE
        persistent_store.set_max_usearch_label(90)
        assert persistent_store.get_max_usearch_label() == 90