"""Unit tests for the ONNX query embedder and runtime selection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from simgrep.adapters.onnx_embedder import OnnxEmbedder, _load_tokenizer, _truncation_limit, cached_hub_file, onnx_cache_dir
from simgrep.models import AppConfig
from simgrep.runtime import LazyBulkEmbedder, RuntimeFactory


class _FakeTokenizer:
    """Splits on whitespace; token ids are word lengths so the fake session can
    produce deterministic per-position vectors."""

    def encode(self, text: str) -> Any:
        ids = [len(word) + 1 for word in text.split()] or [1]
        return type("Encoding", (), {"ids": ids})()


class _FakeSession:
    """last_hidden_state[b, t, :] = token_id, so mean pooling over the real
    attention mask is exactly the mean token id (padded zeros excluded)."""

    def __init__(self, ndim: int = 4) -> None:
        self.ndim = ndim
        self.calls: list[dict[str, np.ndarray]] = []

    def run(self, _outputs: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append(inputs)
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        hidden = np.zeros((ids.shape[0], ids.shape[1], self.ndim), dtype=np.float32)
        for b in range(ids.shape[0]):
            for t in range(ids.shape[1]):
                hidden[b, t, ids[b, t] % self.ndim] = float(ids[b, t])
                hidden[b, t, (ids[b, t] + 1) % self.ndim] = 1.0
        return [hidden * mask[:, :, None]]


def _embedder(session: _FakeSession) -> OnnxEmbedder:
    return OnnxEmbedder(
        "fake/model",
        session=session,
        tokenizer=_FakeTokenizer(),
        ndim=session.ndim,
        pooling="mean",
    )


def test_encode_mean_pools_over_real_tokens_and_normalizes() -> None:
    session = _FakeSession()
    embedder = _embedder(session)
    vectors = embedder.encode(["one two three"])
    assert vectors.shape == (1, 4)
    # mean token id of [3+1=4? no: words 'one'(3)+1=4,'two'(3)+1=4,'three'(5)+1=6] -> (4+4+6)/3
    expected = np.zeros(4)
    for tid in (4, 4, 6):
        expected[tid % 4] += tid
        expected[(tid + 1) % 4] += 1.0
    expected /= 3.0
    expected /= np.linalg.norm(expected)
    assert np.allclose(vectors[0], expected, atol=1e-6)


def test_encode_empty_batch_returns_zero_rows() -> None:
    embedder = _embedder(_FakeSession())
    vectors = embedder.encode([])
    assert vectors.shape == (0, 4)


def test_encode_scatters_length_sorted_batches_back_to_input_order() -> None:
    session = _FakeSession()
    embedder = _embedder(session)
    texts = ["a b c d e f g h", "short one", "x"]
    vectors = embedder.encode(texts, batch_size=1)
    singles = np.stack([embedder.encode([t])[0] for t in texts])
    assert np.allclose(vectors, singles, atol=1e-6)


def test_cls_pooling_uses_first_position() -> None:
    session = _FakeSession()
    embedder = OnnxEmbedder("fake/model", session=session, tokenizer=_FakeTokenizer(), ndim=4, pooling="cls")
    vectors = embedder.encode(["one two"])
    first = session.calls[0]["input_ids"][0, 0]
    expected = np.zeros(4)
    expected[first % 4] += first
    expected[(first + 1) % 4] += 1.0
    expected /= np.linalg.norm(expected)
    assert np.allclose(vectors[0], expected, atol=1e-6)


def test_query_prefix_is_prepended_once() -> None:
    session = _FakeSession()
    seen: list[str] = []

    class _RecordingTokenizer(_FakeTokenizer):
        def encode(self, text: str) -> Any:
            seen.append(text)
            return super().encode(text)

    embedder = OnnxEmbedder(
        "fake/model",
        session=session,
        tokenizer=_RecordingTokenizer(),
        ndim=4,
        pooling="mean",
        query_prompt="Query: ",
    )
    embedder.encode(["hello"], is_query=True)
    embedder.encode(["hello"], is_query=False)
    assert seen == ["Query: hello", "hello"]


def test_missing_cache_raises_informative_error(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="ONNX"):
        OnnxEmbedder("fake/model", cache_root=tmp_path)


def test_loads_from_cache_dir(tmp_path: Path) -> None:
    cache = onnx_cache_dir("fake/model", root=tmp_path)
    cache.mkdir(parents=True)
    session = _FakeSession()
    # a real session cannot be serialized; the loader only needs the file to exist
    # for validation, so write a placeholder and inject the session for the run.
    (cache / "model.onnx").write_bytes(b"onnx")
    (cache / "meta.json").write_text(json.dumps({"model": "fake/model", "ndim": 4, "pooling": "mean"}))
    embedder = OnnxEmbedder("fake/model", cache_root=tmp_path, session=session, tokenizer=_FakeTokenizer())
    assert embedder.ndim == 4
    assert embedder._pooling == "mean"


def test_factory_torch_mode_materializes_eagerly(monkeypatch: Any) -> None:
    """Torch fallbacks must materialize at factory time: a lazy torch query embedder
    would import torch after usearch loads and trip the OpenMP segfault."""
    constructed: list[str] = []

    class FakeInner:
        ndim = 5

        def __init__(self, model_name: str, normalize_embeddings: bool) -> None:
            constructed.append(model_name)

    monkeypatch.setenv("SIMGREP_EMBED_RUNTIME", "torch")
    monkeypatch.setattr("simgrep.runtime.SentenceEmbedder", FakeInner)
    runtime = RuntimeFactory().for_app(AppConfig(model="m"))
    assert isinstance(runtime.embedder, LazyBulkEmbedder)
    assert isinstance(runtime.query_embedder, FakeInner)
    assert constructed == ["m"], "torch mode must materialize before any index load"


def test_factory_broken_onnx_export_falls_back_on_first_use(monkeypatch: Any) -> None:
    constructed: list[str] = []

    class FakeInner:
        ndim = 5

        def __init__(self, model_name: str, normalize_embeddings: bool) -> None:
            constructed.append(model_name)

    def broken_exporter(embedder: Any, model: str) -> None:
        raise RuntimeError("export failed")

    monkeypatch.delenv("SIMGREP_EMBED_RUNTIME", raising=False)
    monkeypatch.setattr("simgrep.runtime.SentenceEmbedder", FakeInner)
    monkeypatch.setattr("simgrep.runtime.has_onnx_cache", lambda model: False)
    monkeypatch.setattr("simgrep.runtime._build_onnx_query_embedder", broken_exporter)
    runtime = RuntimeFactory().for_app(AppConfig(model="m"))
    from simgrep import runtime as rt

    assert isinstance(runtime.query_embedder, rt._LazyQueryEmbedder)
    assert constructed == [], "nothing materializes at factory time"
    runtime.query_embedder.ndim
    assert isinstance(runtime.query_embedder._inner, FakeInner)
    assert constructed == ["m"], "broken export must fall back to torch on first use"


def _cache_fake_hf_model(tmp_path: Path, monkeypatch: Any, model_max_length: int) -> None:
    """Materialize a fake HF hub cache entry with a distinct model_max_length."""
    from tokenizers import Tokenizer, models, pre_tokenizers  # type: ignore[import-untyped]

    snapshot = tmp_path / "models--fake--model" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    tok = Tokenizer(models.WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(snapshot / "tokenizer.json"))
    (snapshot / "tokenizer_config.json").write_text(json.dumps({"model_max_length": model_max_length}))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))


def test_load_tokenizer_truncation_follows_model_max_length(tmp_path: Path, monkeypatch: Any) -> None:
    """The ONNX query path must truncate like the bulk torch path: at the
    tokenizer's own model_max_length, never a hardcoded 512."""
    _cache_fake_hf_model(tmp_path, monkeypatch, model_max_length=8192)
    tokenizer = _load_tokenizer("fake/model")
    assert tokenizer.truncation is not None
    assert tokenizer.truncation["max_length"] == 8192


def test_load_tokenizer_guards_unbounded_sentinel(tmp_path: Path, monkeypatch: Any) -> None:
    """Tokenizers reporting ~1e30 must fall back to a finite cap."""
    _cache_fake_hf_model(tmp_path, monkeypatch, model_max_length=10**30)
    tokenizer = _load_tokenizer("fake/model")
    assert tokenizer.truncation is not None
    assert tokenizer.truncation["max_length"] < 1_000_000


def test_cached_hub_file_prefers_refs_main_snapshot(tmp_path: Path, monkeypatch: Any) -> None:
    """After a model update leaves two cached snapshots, resolution must follow
    refs/main like huggingface_hub, not the newest-mtime snapshot."""
    base = tmp_path / "models--fake--model"
    old_snap = base / "snapshots" / "oldoldold"
    new_snap = base / "snapshots" / "newnewnew"
    for snap, limit in ((old_snap, 8192), (new_snap, 4096)):
        snap.mkdir(parents=True)
        (snap / "tokenizer_config.json").write_text(json.dumps({"model_max_length": limit}))
    # Unambiguous mtime ordering: the refs/main-pinned snapshot is the older one.
    os.utime(old_snap / "tokenizer_config.json", (1_000_000_000, 1_000_000_000))
    os.utime(new_snap / "tokenizer_config.json", (2_000_000_000, 2_000_000_000))
    (base / "refs").mkdir()
    (base / "refs" / "main").write_text("oldoldold\n")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    resolved = cached_hub_file("fake/model", "tokenizer_config.json")
    assert resolved is not None
    assert resolved.parent.name == "oldoldold"
    assert _truncation_limit("fake/model") == 8192
