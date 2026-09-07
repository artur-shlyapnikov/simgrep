from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simgrep.adapters.embedder import SentenceEmbedder
from simgrep.adapters.onnx_embedder import OnnxEmbedder, has_onnx_cache, onnx_cache_dir

pytestmark = pytest.mark.external

MODEL = "ibm-granite/granite-embedding-30m-english"
TEXTS = [
    "def search_index(query, k):\n    return index.search(query, k)",
    "The vector store persists embeddings next to the metadata database.",
    "class RuntimeFactory:\n    def for_project(self, config): ...",
    "Configuration values are validated before the CLI dispatches a command.",
    "Chunk overlap keeps sentences readable across chunk boundaries.",
]
QUERIES = ["vector index save load", "config validation error", "chunk overlap tokens"]


def _torch_embedder() -> SentenceEmbedder:
    return SentenceEmbedder(MODEL, normalize_embeddings=True)


def test_onnx_export_and_query_parity(tmp_path: Path) -> None:
    torch_embedder = _torch_embedder()
    cache = onnx_cache_dir(MODEL, root=tmp_path)
    torch_embedder.export_onnx(cache)
    assert has_onnx_cache(MODEL, root=tmp_path)

    onnx_embedder = OnnxEmbedder(MODEL, cache_root=tmp_path)
    assert onnx_embedder.ndim == torch_embedder.ndim

    doc_vectors = torch_embedder.encode(TEXTS)
    onnx_vectors = onnx_embedder.encode(TEXTS)
    cosine = 1.0 - np.sum(doc_vectors.astype(np.float64) * onnx_vectors, axis=1)
    assert float(cosine.max()) < 1e-5, f"cosine delta too large: {cosine.max():.2e}"

    for query in QUERIES:
        torch_top = set(np.argsort(-(doc_vectors @ torch_embedder.encode([query])[0]))[:5].tolist())
        onnx_top = set(np.argsort(-(onnx_vectors @ onnx_embedder.encode([query], is_query=True)[0]))[:5].tolist())
        assert torch_top == onnx_top, f"top-5 ranking diverged for query {query!r}: {torch_top} vs {onnx_top}"


def test_fp16_encode_matches_fp32() -> None:
    import os

    os.environ["SIMGREP_ENCODE_DTYPE"] = "fp32"
    try:
        reference_embedder = SentenceEmbedder(MODEL, normalize_embeddings=True)
    finally:
        os.environ.pop("SIMGREP_ENCODE_DTYPE", None)
    assert not reference_embedder._use_fp16
    reference = reference_embedder.encode(TEXTS)

    fp16_embedder = SentenceEmbedder(MODEL, normalize_embeddings=True)
    if not fp16_embedder._use_fp16:
        pytest.skip("fp16 fast path not active on this device (MPS unavailable)")
    fp16_vectors = fp16_embedder.encode(TEXTS)
    cosine = 1.0 - np.sum(reference.astype(np.float64) * fp16_vectors, axis=1)
    assert float(cosine.max()) < 1e-5, f"fp16 cosine delta too large: {cosine.max():.2e}"
