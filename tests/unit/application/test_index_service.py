import pathlib

import numpy as np
import pytest

from simgrep.core.abstractions import Embedder, Repository, TextExtractor, TokenChunker
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
