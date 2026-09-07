from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simgrep.adapters.chunker import HFChunker
from simgrep.adapters.embedder import SentenceEmbedder
from simgrep.adapters.extractor import TextExtractor
from simgrep.adapters.onnx_embedder import OnnxEmbedder, has_onnx_cache, onnx_cache_dir
from simgrep.adapters.vector import USearchIndex
from simgrep.errors import SimgrepError
from simgrep.models import AppConfig, ProjectConfig


@dataclass(frozen=True)
class Runtime:
    extractor: TextExtractor
    chunker: Any
    embedder: Any
    query_embedder: Any = None
    reranker: Any = None

    def __post_init__(self) -> None:
        if self.query_embedder is None:
            object.__setattr__(self, "query_embedder", self.embedder)

    def require_bulk(self) -> None:
        """Materialize the torch embedder and tokenizer chunker before indexing.

        Call sites are the bulk entry points only; this keeps the torch libomp
        load ahead of the first USearchIndex (OpenMP segfault guard).
        """
        self.embedder.require()
        self.chunker.require()

    def ensure_query_runtime(self) -> None:
        """Materialize the query embedder ahead of any USearchIndex construction.

        With no local ONNX cache, the lazy query embedder falls back to torch
        mid-request; resolving it first keeps torch's libomp load ahead of
        usearch on that fallback. An ONNX cache hit stays import-free.
        """
        require = getattr(self.query_embedder, "require", None)
        if callable(require):
            require()

    def new_vector_index(self, ndim: int) -> USearchIndex:
        # Every USearchIndex in the process is born here, so resolve the lazy
        # query embedder first: an absent ONNX cache falls back to torch
        # mid-request, and that torch libomp load must precede usearch's.
        self.ensure_query_runtime()
        return USearchIndex(ndim=ndim, metric="ip")


def _onnx_meta_ndim(model: str) -> int | None:
    """Embedding dimension from the local ONNX cache meta, if exported."""
    try:
        meta_path = onnx_cache_dir(model) / "meta.json"
        if not meta_path.is_file():
            return None
        ndim = int(json.loads(meta_path.read_text(encoding="utf-8")).get("ndim") or 0)
        return ndim or None
    except Exception:
        return None


def _ndim_without_torch(model: str) -> int | None:
    """Embedding dimension from the hub-cached config.json, no torch import."""
    try:
        from simgrep.adapters.onnx_embedder import cached_hub_file

        path = cached_hub_file(model, "config.json")
        if path is None:
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(model, "config.json", local_files_only=True))
        hidden = int(json.loads(path.read_text(encoding="utf-8")).get("hidden_size") or 0)
        return hidden or None
    except Exception:
        return None


def _build_onnx_query_embedder(embedder: Any, model: str) -> OnnxEmbedder:
    if not has_onnx_cache(model):
        embedder.require().export_onnx(onnx_cache_dir(model))
    return OnnxEmbedder(model)


def query_runtime_mode() -> str:
    return os.environ.get("SIMGREP_EMBED_RUNTIME", "auto").strip().lower() or "auto"


def assert_safe_bulk_entry() -> None:
    """Guard the torch-before-usearch libomp invariant on long-lived servers.

    Serving a query loads usearch (libomp resident, torch unloaded); a later
    in-process bulk index imports torch afterwards, which can segfault unless
    OMP_NUM_THREADS=1 (set process-wide by ``simgrep/__init__.py`` — the
    documented mitigation). CLI one-shot invocations are unaffected: a fresh
    process reaches ``_require_bulk`` before any usearch import. Cheap check:
    no torch/usearch import happens here.
    """
    from simgrep.adapters import vector as vector_mod

    if vector_mod._USEARCH_MODULE is None or "torch" in sys.modules:
        return
    if os.environ.get("OMP_NUM_THREADS") == "1":
        return
    raise SimgrepError(
        "unsafe bulk index entry: usearch (libomp) is already loaded but torch "
        "is not, and OMP_NUM_THREADS!=1 so the documented mitigation does not "
        "apply. Restart the server, set SIMGREP_EMBED_RUNTIME=torch, or run "
        "the index via the simgrep CLI (fresh subprocess) instead."
    )


class LazyBulkEmbedder:
    """Defers the torch/SentenceEmbedder construction until bulk encode.

    Query-only CLI flows (search/similar over a fresh persistent index)
    never encode documents, so they must never pay the torch import. Anything
    that will encode documents goes through :meth:`require`, which marks torch
    pending so the usearch OpenMP guard keeps the safe libomp order.
    """

    def __init__(self, model_name: str, normalize_embeddings: bool = True) -> None:
        self._model_name = model_name
        self._normalize_embeddings = normalize_embeddings
        self._inner: Any = None

    def _mark_pending(self) -> None:
        from simgrep.adapters import vector as vector_mod

        vector_mod.mark_torch_pending()

    def require(self) -> Any:
        if self._inner is None:
            self._mark_pending()
            self._inner = SentenceEmbedder(self._model_name, normalize_embeddings=self._normalize_embeddings)
            from simgrep.adapters import vector as vector_mod

            vector_mod.mark_torch_loaded()
        return self._inner

    @property
    def ndim(self) -> int:
        if self._inner is not None:
            return int(self._inner.ndim)
        # Query-only flows hit this property (e.g. persistent reader sizing the
        # index). Prefer the local ONNX meta, then the hub-cached config —
        # both import-free — before materializing torch.
        meta = _onnx_meta_ndim(self._model_name)
        if meta is not None:
            return meta
        ndim = _ndim_without_torch(self._model_name)
        if ndim is not None:
            return int(ndim)
        return int(self.require().ndim)

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int | None = None) -> Any:
        return self.require().encode(texts, is_query=is_query, batch_size=batch_size)


class LazyChunker:
    """Defers the transformers tokenizer load until files are actually chunked.

    Query-only CLI flows never chunk, and AutoTokenizer.from_pretrained pulls
    in transformers and torch, so the chunker must be as lazy as the embedder.
    """

    def __init__(self, model_name: str, chunk_size: int, overlap: int) -> None:
        self._args = (model_name, chunk_size, overlap)
        self._inner: Any = None

    def require(self) -> Any:
        if self._inner is None:
            self._inner = HFChunker(model_name=self._args[0], chunk_size=self._args[1], overlap=self._args[2])
        return self._inner

    def chunk(self, text: str) -> Any:
        return self.require().chunk(text)


class _LazyQueryEmbedder:
    """Builds the ONNX query session on first attribute use.

    Daemon-served queries never encode locally, so deferring the ~150ms
    ONNX session + tokenizer build keeps that cost off the client path;
    in-process flows build it at the first encode instead of at startup.
    """

    def __init__(self, bulk: Any, model: str) -> None:
        self._bulk = bulk
        self._model = model
        self._inner: Any = None
        self._lock = threading.Lock()

    def require(self) -> Any:
        with self._lock:
            if self._inner is None:
                try:
                    self._inner = _build_onnx_query_embedder(self._bulk, self._model)
                except Exception:
                    self._inner = self._bulk.require()
            return self._inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.require(), name)


class RuntimeFactory:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, int], Runtime] = {}

    def for_project(self, config: ProjectConfig) -> Runtime:
        return self._runtime(config.model, config.chunk_size, config.chunk_overlap)

    def for_app(self, config: AppConfig) -> Runtime:
        return self._runtime(config.model, config.chunk_size, config.chunk_overlap)

    def _runtime(self, model: str, chunk_size: int, chunk_overlap: int) -> Runtime:
        key = (model, chunk_size, chunk_overlap)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        embedder = LazyBulkEmbedder(model, normalize_embeddings=True)
        # Ordering rule: if torch may load in this process, it must load BEFORE
        # the first USearchIndex (OpenMP libomp ownership). The ONNX query path
        # never imports torch, so it defers the session build to first use;
        # every torch fallback materializes eagerly here, ahead of any index load.
        if query_runtime_mode() == "torch":
            query_embedder = embedder.require()
        else:
            query_embedder = _LazyQueryEmbedder(embedder, model)
        runtime = Runtime(
            extractor=TextExtractor(),
            chunker=LazyChunker(model_name=model, chunk_size=chunk_size, overlap=chunk_overlap),
            embedder=embedder,
            query_embedder=query_embedder,
        )
        self._cache[key] = runtime
        return runtime
