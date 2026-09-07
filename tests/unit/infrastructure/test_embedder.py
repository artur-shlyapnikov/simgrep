from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from simgrep.adapters.embedder import SentenceEmbedder
from simgrep.errors import SimgrepError


class _Output:
    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


def _hidden_from_input_ids(input_ids: torch.Tensor, dim: int) -> torch.Tensor:
    """Deterministic 'model': hidden rows derive from input ids."""
    base = input_ids.unsqueeze(-1).float().expand(*input_ids.shape, dim).clone()
    base[..., 0] = base[..., 0] + 1.0
    return base


def _make_tokenizer() -> MagicMock:
    tokenizer = MagicMock()

    def call(texts: list[str], padding: object = None, truncation: object = None, return_tensors: object = None) -> dict[str, torch.Tensor]:
        del padding, truncation, return_tensors
        ids = torch.arange(len(texts), dtype=torch.long).unsqueeze(1).repeat(1, 4) + 5
        mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": mask}

    tokenizer.side_effect = call
    return tokenizer


def _make_model(dim: int, forward: Callable[..., Any] | None = None) -> MagicMock:
    model = MagicMock()
    model.config.hidden_size = dim
    model.to.return_value = model
    model.eval.return_value = model
    if forward is None:
        forward = lambda **features: _Output(_hidden_from_input_ids(features["input_ids"], dim))  # noqa: E731
    model.side_effect = forward
    return model


def _fake_download_factory(repo_files: dict[str, str] | None) -> Callable[..., str]:
    """hf_hub_download contract: return a path to an existing cached file."""

    def fake_download(repo_id: str, filename: str, **kwargs: object) -> str:
        del repo_id, kwargs
        if repo_files and filename in repo_files:
            return repo_files[filename]
        raise FileNotFoundError(filename)

    return fake_download


def _spy_tokenizer(captured: list[list[str]]) -> MagicMock:
    tokenizer = _make_tokenizer()
    base_call = cast(Callable[..., dict[str, torch.Tensor]], tokenizer.side_effect)

    def spy_call(texts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        captured.append(list(texts))
        return base_call(texts, **kwargs)

    tokenizer.side_effect = spy_call
    return tokenizer


def _build_embedder(
    monkeypatch: pytest.MonkeyPatch,
    model: MagicMock,
    repo_files: dict[str, str] | None = None,
    model_name: str = "test-model",
    normalize_embeddings: bool = True,
    tokenizer: MagicMock | None = None,
) -> SentenceEmbedder:
    monkeypatch.setenv("SIMGREP_DEVICE", "cpu")
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer or _make_tokenizer()),
        patch("transformers.AutoModel.from_pretrained", return_value=model),
        patch("huggingface_hub.hf_hub_download", side_effect=_fake_download_factory(repo_files)),
    ):
        return SentenceEmbedder(model_name, normalize_embeddings=normalize_embeddings)


def test_encode_empty_list_returns_2d_empty_array_with_ndim(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_model(3)
    embedder = _build_embedder(monkeypatch, model)
    embeddings = embedder.encode([])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0, 3)
    assert model.call_count == 0


def test_encode_returns_float32_dtype_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = _build_embedder(monkeypatch, _make_model(2))
    embeddings = embedder.encode(["hello"])
    assert embeddings.dtype == np.float32


def test_normalize_embeddings_true_produces_unit_norm_for_nonzero_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    dim = 2

    def forward(**features: torch.Tensor) -> _Output:
        rows = int(features["input_ids"].shape[0])
        hidden = torch.zeros(rows, 4, dim)
        hidden[:, 0, 0] = 3.0
        hidden[:, 0, 1] = 4.0
        return _Output(hidden)

    embedder = _build_embedder(monkeypatch, _make_model(dim, forward=forward))
    embeddings = embedder.encode(["a", "b", "zero"])
    assert embeddings.shape == (3, 2)
    assert np.allclose(embeddings[0], [0.6, 0.8], atol=1e-6)
    assert np.allclose(embeddings[1], [0.6, 0.8], atol=1e-6)
    assert np.isclose(np.linalg.norm(embeddings[0]), 1.0, atol=1e-6)


def test_normalize_false_keeps_raw_cls_vector(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dim = 2
    modules = tmp_path / "modules.json"
    modules.write_text(MODULES_CLS, encoding="utf-8")
    pooling = tmp_path / "pooling.json"
    pooling.write_text(POOLING_CLS, encoding="utf-8")

    def forward(**features: torch.Tensor) -> _Output:
        rows = int(features["input_ids"].shape[0])
        hidden = torch.zeros(rows, 4, dim)
        hidden[:, 0, 0] = 3.0
        hidden[:, 0, 1] = 4.0
        return _Output(hidden)

    embedder = _build_embedder(
        monkeypatch,
        _make_model(dim, forward=forward),
        repo_files={"modules.json": str(modules), "1_Pooling/config.json": str(pooling)},
        normalize_embeddings=False,
    )
    embeddings = embedder.encode(["a"], is_query=False)
    assert np.allclose(embeddings[0], [3.0, 4.0], atol=1e-6)


def test_batch_size_controls_forward_chunking(monkeypatch: pytest.MonkeyPatch) -> None:
    sizes: list[int] = []

    def forward(**features: torch.Tensor) -> _Output:
        sizes.append(int(features["input_ids"].shape[0]))
        return _Output(_hidden_from_input_ids(features["input_ids"], 2))

    embedder = _build_embedder(monkeypatch, _make_model(2, forward=forward))
    embedder.encode([f"text {i}" for i in range(5)], batch_size=2)
    assert sizes == [2, 2, 1]


def test_length_sorted_batches_preserve_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    dim = 1
    seen: list[int] = []

    def forward(**features: torch.Tensor) -> _Output:
        ids = features["input_ids"][:, 0]
        seen.extend(int(value) for value in ids.tolist())
        return _Output(_hidden_from_input_ids(features["input_ids"], dim))

    embedder = _build_embedder(monkeypatch, _make_model(dim, forward=forward))
    # tokenizer mock assigns id 5+i to text i; short texts must be encoded first
    embeddings = embedder.encode(["aaaa", "b", "cc"], batch_size=8)
    assert sorted(seen) == [5, 6, 7]
    assert embeddings.shape == (3, 1)


def test_qwen_model_prefixes_query_prompt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.json"
    prompt_file.write_text(json.dumps({"prompts": {"query": "Query: "}}), encoding="utf-8")
    captured: list[list[str]] = []
    tokenizer = _spy_tokenizer(captured)

    def fake_download(repo_id: str, filename: str, **kwargs: object) -> str:
        del repo_id, kwargs
        if filename == "config_sentence_transformers.json":
            return str(prompt_file)
        raise FileNotFoundError(filename)

    monkeypatch.setenv("SIMGREP_DEVICE", "cpu")
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.AutoModel.from_pretrained", return_value=_make_model(2)),
        patch("huggingface_hub.hf_hub_download", side_effect=fake_download),
    ):
        embedder = SentenceEmbedder("TestQwen-model")
        embedder.encode(["my query"], is_query=True)
        assert captured == [["Query: my query"]]
        embedder.encode(["plain text"], is_query=False)
        assert captured[-1] == ["plain text"]


def test_non_qwen_model_does_not_prefix_query_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []
    tokenizer = _spy_tokenizer(captured)
    embedder = _build_embedder(monkeypatch, _make_model(2), model_name="granite-model", tokenizer=tokenizer)
    embedder.encode(["my query"], is_query=True)
    assert captured == [["my query"]]


def test_underlying_forward_error_wrapped_in_simgrep_error_with_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**features: torch.Tensor) -> _Output:
        raise RuntimeError("boom")

    embedder = _build_embedder(monkeypatch, _make_model(2, forward=boom), model_name="special-model")
    with pytest.raises(SimgrepError, match="special-model"):
        embedder.encode(["hello"])


def test_ndim_reads_config_without_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_model(384)
    embedder = _build_embedder(monkeypatch, model)
    assert embedder.ndim == 384
    assert model.call_count == 0


def test_missing_hidden_size_raises_instead_of_probing(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_model(384)
    model.config.hidden_size = 0
    monkeypatch.setenv("SIMGREP_DEVICE", "cpu")
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_make_tokenizer()),
        patch("transformers.AutoModel.from_pretrained", return_value=model),
        patch("huggingface_hub.hf_hub_download", side_effect=_fake_download_factory(None)),
    ):
        with pytest.raises(SimgrepError, match="embedding dimension"):
            SentenceEmbedder("some-model")


MODULES_CLS = json.dumps(
    [
        {"type": "sentence_transformers.models.Transformer", "path": ""},
        {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"},
    ]
)
POOLING_CLS = json.dumps({"pooling_mode_cls_token": True, "pooling_mode_mean_tokens": False})
POOLING_MEAN = json.dumps({"pooling_mode_cls_token": False, "pooling_mode_mean_tokens": True})
POOLING_MAX = json.dumps({"pooling_mode_cls_token": False, "pooling_mode_mean_tokens": False, "pooling_mode_max_tokens": True})


def test_pooling_resolution_reads_modules_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def write(filename: str, content: str) -> str:
        target = tmp_path / filename.replace("/", "_")
        target.write_text(content, encoding="utf-8")
        return str(target)

    files_cls = {
        "modules.json": write("modules_cls.json", MODULES_CLS),
        "1_Pooling/config.json": write("pooling_cls.json", POOLING_CLS),
    }
    embedder_cls = _build_embedder(monkeypatch, _make_model(2), repo_files=files_cls)
    assert embedder_cls._pooling == "cls"

    files_mean = {
        "modules.json": write("modules_mean.json", MODULES_CLS),
        "1_Pooling/config.json": write("pooling_mean.json", POOLING_MEAN),
    }
    embedder_mean = _build_embedder(monkeypatch, _make_model(2), repo_files=files_mean, model_name="mean-model")
    assert embedder_mean._pooling == "mean"


def test_pooling_defaults_to_mean_without_modules_json(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = _build_embedder(monkeypatch, _make_model(2), repo_files={})
    assert embedder._pooling == "mean"


def test_unsupported_pooling_mode_fails_loudly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    modules = tmp_path / "modules.json"
    modules.write_text(MODULES_CLS, encoding="utf-8")
    pooling = tmp_path / "pool.json"
    pooling.write_text(POOLING_MAX, encoding="utf-8")

    def fake_download(repo_id: str, filename: str, **kwargs: object) -> str:
        del repo_id, kwargs
        return {"modules.json": str(modules), "1_Pooling/config.json": str(pooling)}[filename]

    monkeypatch.setenv("SIMGREP_DEVICE", "cpu")
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_make_tokenizer()),
        patch("transformers.AutoModel.from_pretrained", return_value=_make_model(2)),
        patch("huggingface_hub.hf_hub_download", side_effect=fake_download),
    ):
        with pytest.raises(SimgrepError, match="Unsupported pooling mode"):
            SentenceEmbedder("max-model")
