import pathlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from simgrep.core.abstractions import (
    Embedder,
    Repository,
    TextExtractor,
    TokenChunker,
    VectorIndex,
)
from simgrep.services.index_service import IndexService


@pytest.fixture
def index_service(
    fake_text_extractor: TextExtractor,
    fake_token_chunker: TokenChunker,
    fake_embedder: Embedder,
    fake_repository: Repository,
    fake_vector_index_factory,
) -> IndexService:
    vector_index = fake_vector_index_factory(ndim=fake_embedder.ndim)
    return IndexService(
        extractor=fake_text_extractor,
        chunker=fake_token_chunker,
        embedder=fake_embedder,
        store=fake_repository,
        index=vector_index,
    )


def test_index_service_process_file(index_service: IndexService, tmp_path: pathlib.Path) -> None:
    file_path = tmp_path / "test.txt"
    file_path.write_text("This is a test.")
    chunks, embeddings = index_service.process_file(file_path)
    assert len(chunks) > 0
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] == index_service.embedder.ndim


def test_store_file_chunks(index_service: IndexService):
    from simgrep.core.models import Chunk

    chunks = [Chunk(id=0, file_id=1, text="text", start=0, end=4, tokens=1)]
    embeddings = np.random.rand(1, index_service.embedder.ndim).astype(np.float32)

    index_service.store_file_chunks(file_id=1, processed_chunks=chunks, embeddings_np=embeddings)

    assert len(index_service.index) == 1
    assert index_service.final_max_label == 0


def test_run_index_prunes_deleted_files(
    fake_text_extractor: TextExtractor,
    fake_token_chunker: TokenChunker,
    fake_embedder: Embedder,
    tmp_path: pathlib.Path,
) -> None:
    """Test the logic for pruning files that exist in the DB but not on disk."""
    mock_store = MagicMock(spec=Repository)
    mock_index = MagicMock(spec=VectorIndex)
    mock_store.get_max_usearch_label.return_value = None  # for init

    index_service = IndexService(
        extractor=fake_text_extractor,
        chunker=fake_token_chunker,
        embedder=fake_embedder,
        store=mock_store,
        index=mock_index,
    )

    # Arrange
    deleted_file_path = (tmp_path / "deleted.txt").resolve()
    mock_store.get_all_indexed_file_records.return_value = [(1, str(deleted_file_path), "hash123")]
    mock_store.delete_file_records.return_value = [101, 102]

    # Act
    with patch("simgrep.services.index_service.gather_files_to_process", return_value=[]) as mock_gather:
        index_service.run_index(
            target_paths=[tmp_path],
            file_scan_patterns=["*.txt"],
            wipe_existing=False,
            max_workers=1,
            console=MagicMock(),
        )

        # Assert
        mock_gather.assert_called_once_with(tmp_path, ["*.txt"])
        mock_store.get_all_indexed_file_records.assert_called_once()
        mock_store.delete_file_records.assert_called_once_with(1)
        mock_index.remove.assert_called_once()
        np.testing.assert_array_equal(mock_index.remove.call_args[1]["keys"], np.array([101, 102], dtype=np.int64))