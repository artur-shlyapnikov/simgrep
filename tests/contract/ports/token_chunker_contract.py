import pytest
from simgrep.core.abstractions import TokenChunker
from simgrep.core.models import Chunk


@pytest.mark.contract
class TokenChunkerContract:
    def test_chunk_returns_chunk_sequence(self, token_chunker: TokenChunker):
        text = "This is a sample text to be chunked."
        chunks = token_chunker.chunk(text)
        assert isinstance(chunks, list)
        if chunks:
            assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_empty_text(self, token_chunker: TokenChunker):
        assert token_chunker.chunk("") == []
        assert token_chunker.chunk("   ") == []
