from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from transformers import PreTrainedTokenizerBase

from simgrep.adapters.hf_chunker import HFChunker
from simgrep.adapters.sentence_embedder import SentenceEmbedder


@pytest.mark.external
class TestHFTokenChunker:
    def test_empty_text(self, hf_chunker: HFChunker) -> None:
        assert hf_chunker.chunk("") == []

    def test_text_shorter_than_chunk_size(self, hf_chunker: HFChunker) -> None:
        text = "This is a short text."
        chunks = hf_chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].text.lower() == text.lower()

    def test_text_that_encodes_to_nothing(self, hf_chunker: HFChunker) -> None:
        # A zero-width space might be ignored by the tokenizer, resulting in no tokens.
        text_that_encodes_to_nothing = "\u200b"
        chunks = hf_chunker.chunk(text_that_encodes_to_nothing)
        assert chunks == []


@pytest.mark.external
class TestSentenceEmbedder:
    def test_generate_valid_embeddings(self, hf_embedder: SentenceEmbedder) -> None:
        import numpy as np

        texts = ["Hello world", "Simgrep is amazing"]
        embeddings = hf_embedder.encode(texts)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == hf_embedder.ndim

    def test_generate_embeddings_empty_list(self, hf_embedder: SentenceEmbedder) -> None:
        import numpy as np

        texts = []
        embeddings = hf_embedder.encode(texts)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 0

    def test_generate_embeddings_invalid_model(self) -> None:
        with pytest.raises(Exception):
            SentenceEmbedder("this-model-does-not-exist-ever-12345")

    def test_qwen_query_prompt(self) -> None:
        with patch("simgrep.adapters.sentence_embedder.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[0.1, 0.2]], dtype=np.float32)
            mock_st.return_value = mock_model

            # Use a model name that contains "qwen"
            embedder = SentenceEmbedder("qwen-test-model")
            embedder.encode(["my query"], is_query=True)

            mock_model.encode.assert_called_once()
            _, call_kwargs = mock_model.encode.call_args
            assert call_kwargs.get("prompt_name") == "query"

    def test_non_qwen_no_query_prompt(self, hf_embedder: SentenceEmbedder) -> None:
        # hf_embedder uses 'all-MiniLM-L6-v2' which is not a qwen model
        with patch.object(hf_embedder._model, "encode", wraps=hf_embedder._model.encode) as spy_encode:
            hf_embedder.encode(["my query"], is_query=True)
            spy_encode.assert_called_once()
            _, call_kwargs = spy_encode.call_args
            assert "prompt_name" not in call_kwargs