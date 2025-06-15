import pytest
from simgrep.adapters.hf_chunker import HFChunker
from simgrep.adapters.sentence_embedder import SentenceEmbedder
from simgrep.adapters.unstructured_extractor import UnstructuredExtractor
from simgrep.adapters.usearch_index import USearchIndex
from simgrep.repository import MetadataStore
from simgrep.core.abstractions import Embedder, Repository, TextExtractor, TokenChunker, VectorIndex
import pathlib

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


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
def metadata_store(tmp_path: pathlib.Path) -> Repository:
    db_path = tmp_path / "test_metadata.duckdb"
    store = MetadataStore(persistent=True, db_path=db_path)
    yield store
    store.close()


# These fixtures are used by pytest_generate_tests to select which adapter to test
@pytest.fixture
def embedder(request) -> Embedder:
    return request.getfixturevalue(request.param)


@pytest.fixture
def text_extractor(request) -> TextExtractor:
    return request.getfixturevalue(request.param)


@pytest.fixture
def token_chunker(request) -> TokenChunker:
    return request.getfixturevalue(request.param)


@pytest.fixture
def vector_index(request) -> VectorIndex:
    return request.getfixturevalue(request.param)


@pytest.fixture
def repository(request) -> Repository:
    return request.getfixturevalue(request.param)
