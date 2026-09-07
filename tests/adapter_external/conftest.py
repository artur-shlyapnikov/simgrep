import pytest

from simgrep.adapters.chunker import HFChunker
from simgrep.adapters.embedder import SentenceEmbedder
from simgrep.adapters.extractor import TextExtractor

MODEL_NAME = "ibm-granite/granite-embedding-30m-english"


@pytest.fixture(scope="session")
def hf_embedder() -> SentenceEmbedder:
    return SentenceEmbedder(MODEL_NAME)


@pytest.fixture(scope="session")
def hf_chunker() -> HFChunker:
    return HFChunker(MODEL_NAME, chunk_size=128, overlap=20)


@pytest.fixture(scope="session")
def text_extractor() -> TextExtractor:
    return TextExtractor()
