import pathlib
import pytest
from simgrep.core.abstractions import Repository


@pytest.mark.contract
class RepositoryContract:
    def test_can_insert_and_retrieve_file_and_chunks(self, repository: Repository, tmp_path: pathlib.Path):
        # This is a simplified test. A full contract would test each method.
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")

        # Clear any previous data
        repository.clear_persistent_project_data()

        file_id = repository.insert_indexed_file_record(
            file_path=str(file_path),
            content_hash="123",
            file_size_bytes=7,
            last_modified_os_timestamp=file_path.stat().st_mtime,
        )
        assert file_id is not None

        chunk_records = [
            {
                "file_id": file_id,
                "usearch_label": 1,
                "chunk_text": "content",
                "start_char_offset": 0,
                "end_char_offset": 7,
                "token_count": 1,
                "embedding_hash": None,
            }
        ]
        repository.batch_insert_text_chunks(chunk_records)

        retrieved = repository.retrieve_filtered_chunk_details(usearch_labels=[1])
        assert len(retrieved) == 1
        assert retrieved[0]["chunk_text"] == "content"

    def test_get_index_counts(self, repository: Repository):
        repository.clear_persistent_project_data()
        files, chunks = repository.get_index_counts()
        assert isinstance(files, int)
        assert isinstance(chunks, int)
        assert files == 0
        assert chunks == 0
