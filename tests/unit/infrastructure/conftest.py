import pathlib
from typing import Iterator, cast

import pytest

from simgrep.adapters.hf_chunker import HFChunker
from simgrep.adapters.sentence_embedder import SentenceEmbedder
from simgrep.adapters.unstructured_extractor import UnstructuredExtractor
from simgrep.adapters.usearch_index import USearchIndex
from simgrep.core.abstractions import (
    Embedder,
    Repository,
    TextExtractor,
    TokenChunker,
    VectorIndex,
)
from simgrep.repository import MetadataStore

MODEL_NAME = "ibm-granite/granite-embedding-30m-english"


@pytest.fixture(scope="session")
def hf_embedder() -> SentenceEmbedder:
    return SentenceEmbedder(MODEL_NAME)


@pytest.fixture(scope="session")
def unstructured_extractor() -> UnstructuredExtractor:
    return UnstructuredExtractor()


@pytest.fixture(scope="session")
def hf_chunker() -> HFChunker:
    return HFChunker(MODEL_NAME, chunk_size=128, overlap=20)


@pytest.fixture
def usearch_index(hf_embedder: Embedder) -> USearchIndex:
    return USearchIndex(ndim=hf_embedder.ndim)


@pytest.fixture
def metadata_store(tmp_path: pathlib.Path) -> Iterator[Repository]:
    db_path = tmp_path / "test_metadata.duckdb"
    store = MetadataStore(persistent=True, db_path=db_path)
    yield store
    store.close()


# These fixtures are used by pytest_generate_tests to select which adapter to test
@pytest.fixture
def embedder(request: pytest.FixtureRequest) -> Embedder:
    return cast(Embedder, request.getfixturevalue(request.param))


@pytest.fixture
def text_extractor(request: pytest.FixtureRequest) -> TextExtractor:
    return cast(TextExtractor, request.getfixturevalue(request.param))


@pytest.fixture
def token_chunker(request: pytest.FixtureRequest) -> TokenChunker:
    return cast(TokenChunker, request.getfixturevalue(request.param))


@pytest.fixture
def vector_index(request: pytest.FixtureRequest) -> VectorIndex:
    return cast(VectorIndex, request.getfixturevalue(request.param))


@pytest.fixture
def repository(request: pytest.FixtureRequest) -> Repository:
    return cast(Repository, request.getfixturevalue(request.param))
