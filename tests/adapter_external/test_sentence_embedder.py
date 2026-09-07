from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from simgrep.adapters.embedder import SentenceEmbedder

pytestmark = pytest.mark.external


class TestSentenceEmbedder:
    def test_generate_valid_embeddings(self, hf_embedder: SentenceEmbedder) -> None:
        texts = ["Hello world", "Simgrep is amazing"]
        embeddings = hf_embedder.encode(texts)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == hf_embedder.ndim

    def test_generate_embeddings_empty_list(self, hf_embedder: SentenceEmbedder) -> None:
        embeddings = hf_embedder.encode([])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 0

    def test_generate_embeddings_invalid_model(self) -> None:
        with pytest.raises(Exception):
            SentenceEmbedder("this-model-does-not-exist-ever-12345")

    def test_qwen_query_prompt(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A qwen-style query prompt from config_sentence_transformers.json prefixes query texts."""
        prompt_file = tmp_path / "prompt.json"
        prompt_file.write_text('{"prompts": {"query": "Instruct: x\\nQuery: "}}', encoding="utf-8")
        captured: list[list[str]] = []
        tokenizer = MagicMock()
        base_call = MagicMock(
            side_effect=lambda texts, **kwargs: {"input_ids": torch.arange(len(texts)).unsqueeze(1), "attention_mask": torch.ones(len(texts), 1)}
        )

        def spy_call(texts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
            captured.append(list(texts))
            return cast("dict[str, torch.Tensor]", base_call(texts, **kwargs))

        tokenizer.side_effect = spy_call
        model = MagicMock()
        model.config.hidden_size = 2
        model.to.return_value = model
        model.eval.return_value = model
        model.side_effect = lambda **features: MagicMock(last_hidden_state=torch.ones(len(features["input_ids"]), 1, 2))

        def fake_download(repo_id: str, filename: str, **kwargs: object) -> str:
            del repo_id, kwargs
            if filename == "config_sentence_transformers.json":
                return str(prompt_file)
            raise FileNotFoundError(filename)

        monkeypatch.setenv("SIMGREP_DEVICE", "cpu")
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch("transformers.AutoModel.from_pretrained", return_value=model),
            patch("huggingface_hub.hf_hub_download", side_effect=fake_download),
        ):
            embedder = SentenceEmbedder("qwen-test-model")
            embedder.encode(["my query"], is_query=True)

        assert captured == [["Instruct: x\nQuery: my query"]]

    def test_non_qwen_no_query_prompt(self, hf_embedder: SentenceEmbedder) -> None:
        assert hf_embedder._query_prefix() == ""
        captured: list[list[str]] = []
        base_call = hf_embedder._tokenizer

        def spy_call(texts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
            captured.append(list(texts))
            return cast("dict[str, torch.Tensor]", base_call(texts, **kwargs))

        hf_embedder._tokenizer = spy_call
        try:
            hf_embedder.encode(["my query"], is_query=True)
        finally:
            hf_embedder._tokenizer = base_call
        assert captured == [["my query"]]

    def test_encode_accepts_batch_size_parameter(self, hf_embedder: SentenceEmbedder) -> None:
        texts = [f"test text {i}" for i in range(10)]
        result = hf_embedder.encode(texts, batch_size=4)
        assert result.shape == (10, hf_embedder.ndim)

    def test_ndim_uses_model_metadata_without_dummy_encode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = MagicMock()
        model.config.hidden_size = 384
        model.to.return_value = model
        model.eval.return_value = model
        monkeypatch.setenv("SIMGREP_DEVICE", "cpu")
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()),
            patch("transformers.AutoModel.from_pretrained", return_value=model),
            patch("huggingface_hub.hf_hub_download", side_effect=FileNotFoundError("none")),
        ):
            embedder = SentenceEmbedder("some-model")

        assert embedder.ndim == 384
        assert model.call_count == 0

    def test_missing_hidden_size_raises_instead_of_probing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = MagicMock()
        model.config.hidden_size = 0
        model.to.return_value = model
        model.eval.return_value = model
        monkeypatch.setenv("SIMGREP_DEVICE", "cpu")
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()),
            patch("transformers.AutoModel.from_pretrained", return_value=model),
            patch("huggingface_hub.hf_hub_download", side_effect=FileNotFoundError("none")),
        ):
            with pytest.raises(Exception, match="embedding dimension"):
                SentenceEmbedder("some-model")
