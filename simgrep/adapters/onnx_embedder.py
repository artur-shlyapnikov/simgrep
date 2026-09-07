"""Query-path embedder on ONNX Runtime.

Truncation parity note: the bulk torch path (:class:`simgrep.adapters.embedder.SentenceEmbedder`)
relies on transformers truncation (``truncation="longest_first"`` with no ``max_length``), which
cuts inputs at the tokenizer's own finite ``model_max_length`` — and not at all when that value
is missing or an unbounded sentinel. The ORT query path instead caps at
``min(model_max_length, 512)`` and falls back to 512 when the config is missing, unreadable, or
sentinel-bounded. That residual divergence is intentional: a bounded sequence length keeps ONNX
Runtime memory allocation predictable on adversarially long queries.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from simgrep.errors import SimgrepError

_META_FILE = "meta.json"
_MODEL_FILE = "model.onnx"
_DEFAULT_TRUNCATION_LIMIT = 512
# Some tokenizers report an unbounded sentinel (~1e30) as model_max_length.
_TRUNCATION_SENTINEL = 1_000_000


def onnx_cache_dir(model_name: str, root: Path | None = None) -> Path:
    slug = hashlib.sha1(model_name.encode(), usedforsecurity=False).hexdigest()[:16]
    base = root if root is not None else Path(os.environ.get("SIMGREP_ONNX_CACHE", Path.home() / ".cache" / "simgrep" / "onnx"))
    return base / slug


def has_onnx_cache(model_name: str, root: Path | None = None) -> bool:
    cache = onnx_cache_dir(model_name, root)
    return (cache / _MODEL_FILE).is_file() and (cache / _META_FILE).is_file()


def _hf_cache_root() -> Path:
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        return Path(hub)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def cached_hub_file(model_name: str, filename: str) -> Path | None:
    """Resolve a file from the local HF hub cache without importing
    huggingface_hub (whose import alone costs ~90ms per invocation).

    Resolution mirrors huggingface_hub: prefer the snapshot pinned by
    ``refs/main``, falling back to newest-mtime across all snapshots when the
    ref is unreadable or does not name an existing snapshot.
    """
    org, _, name = model_name.partition("/")
    slug = f"models--{org}--{name}" if name else f"models--{model_name}"
    model_dir = _hf_cache_root() / slug
    try:
        commit = (model_dir / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        commit = ""
    if commit:
        pinned = model_dir / "snapshots" / commit / filename
        if pinned.is_file():
            return pinned
    snapshots = model_dir / "snapshots"
    try:
        candidates = sorted(snapshots.glob(f"*/{filename}"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return None
    return candidates[0] if candidates else None


def _cached_tokenizer_json(model_name: str) -> Path | None:
    return cached_hub_file(model_name, "tokenizer.json")


def _truncation_limit(model_name: str) -> int:
    """Mirror the bulk torch path: transformers truncates at the tokenizer's own
    ``model_max_length``; fall back to a finite default for unbounded sentinels
    or when the config is unreadable.

    Divergence from the bulk torch path (intentional, bounded ORT memory): the
    bulk path truncates never when ``tokenizer_config.json`` is missing,
    unreadable, or reports the unbounded sentinel, whereas this query path caps
    at :data:`_DEFAULT_TRUNCATION_LIMIT` in those cases — see the module
    docstring.
    """
    config = cached_hub_file(model_name, "tokenizer_config.json")
    if config is None:
        try:
            from huggingface_hub import hf_hub_download

            config = Path(hf_hub_download(model_name, "tokenizer_config.json", local_files_only=True))
        except Exception:
            return _DEFAULT_TRUNCATION_LIMIT
    try:
        limit = int(json.loads(config.read_text(encoding="utf-8")).get("model_max_length") or 0)
    except Exception:
        return _DEFAULT_TRUNCATION_LIMIT
    return limit if 0 < limit < _TRUNCATION_SENTINEL else _DEFAULT_TRUNCATION_LIMIT


def _load_tokenizer(model_name: str) -> Any:
    """Offline-first: never touch the network when the tokenizer is cached."""
    from tokenizers import Tokenizer  # type: ignore[import-untyped]

    cached = _cached_tokenizer_json(model_name)
    if cached is not None:
        tokenizer = Tokenizer.from_file(str(cached))
    else:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(model_name, "tokenizer.json", local_files_only=True)
            tokenizer = Tokenizer.from_file(str(path))
        except Exception:
            tokenizer = Tokenizer.from_pretrained(model_name)
    tokenizer.enable_truncation(max_length=_truncation_limit(model_name))
    return tokenizer


class OnnxEmbedder:
    """Query-path embedder on ONNX Runtime.

    Same seam as :class:`simgrep.adapters.embedder.SentenceEmbedder` (``ndim``,
    ``encode``) but imports neither torch nor transformers, which removes about
    2.5s of import cost from every CLI invocation. Bulk/indexing encode stays
    on the torch-MPS embedder; this class serves the few query vectors per
    invocation, where absolute throughput is irrelevant and startup dominates.
    """

    def __init__(
        self,
        model_name: str,
        normalize_embeddings: bool = True,
        *,
        session: Any = None,
        tokenizer: Any = None,
        ndim: int | None = None,
        pooling: str = "mean",
        query_prompt: str | None = None,
        cache_root: Path | None = None,
    ) -> None:
        self._model_name = model_name
        self._normalize_embeddings = normalize_embeddings
        self._session = session
        self._tokenizer = tokenizer
        self._pooling = pooling
        self._query_prompt = query_prompt
        try:
            cache = onnx_cache_dir(model_name, cache_root)
            if session is None:
                if not has_onnx_cache(model_name, cache_root):
                    raise SimgrepError(
                        f"No ONNX cache for model '{model_name}'.",
                        hint="Run `simgrep index` once (exports the cache), or set SIMGREP_EMBED_RUNTIME=torch.",
                    )
                import onnxruntime as ort  # type: ignore[import-untyped]

                options = ort.SessionOptions()
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session = ort.InferenceSession(str(cache / _MODEL_FILE), options, providers=["CPUExecutionProvider"])
            self._session = session
            self._tokenizer = tokenizer if tokenizer is not None else _load_tokenizer(model_name)
            if ndim is None or (tokenizer is None and session is None):
                if not (cache / _META_FILE).is_file():
                    raise SimgrepError(f"ONNX cache for model '{model_name}' is missing {_META_FILE}.")
                meta = json.loads((cache / _META_FILE).read_text(encoding="utf-8"))
                if meta.get("model") != model_name:
                    raise SimgrepError(f"ONNX cache model mismatch: {meta.get('model')!r} != {model_name!r}.")
                ndim = int(meta["ndim"]) if ndim is None else ndim
                if self._pooling == "mean" and meta.get("pooling") in ("mean", "cls"):
                    self._pooling = str(meta["pooling"])
            self._ndim = int(ndim)
            if self._pooling not in ("mean", "cls"):
                raise SimgrepError(f"Unsupported pooling mode for ONNX embedder: {self._pooling!r}.")
            if self._query_prompt is None:
                self._query_prompt = self._default_query_prompt()
        except SimgrepError:
            raise
        except Exception as exc:
            raise SimgrepError(f"Failed to load ONNX embedder for model '{model_name}'.") from exc

    def _default_query_prompt(self) -> str:
        if "qwen" not in self._model_name.lower():
            return ""
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(self._model_name, "config_sentence_transformers.json", local_files_only=True)
            prompts = json.loads(Path(path).read_text(encoding="utf-8")).get("prompts", {})
            return str(prompts.get("query", ""))
        except Exception:
            return ""

    @property
    def ndim(self) -> int:
        return self._ndim

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int | None = None) -> np.ndarray:
        try:
            if not texts:
                return np.zeros((0, self._ndim), dtype=np.float32)
            if is_query and self._query_prompt:
                texts = [self._query_prompt + text for text in texts]
            size = batch_size if batch_size is not None and batch_size > 0 else 128
            encodings = [self._tokenizer.encode(text).ids for text in texts]
            # Length-sorted batches minimize padding, like the torch embedder.
            order = sorted(range(len(texts)), key=lambda i: len(encodings[i]))
            vectors = np.zeros((len(texts), self._ndim), dtype=np.float32)
            for start in range(0, len(order), size):
                batch_idx = order[start : start + size]
                seq = [encodings[i] for i in batch_idx]
                maxlen = max(len(s) for s in seq)
                ids = np.zeros((len(seq), maxlen), dtype=np.int64)
                mask = np.zeros((len(seq), maxlen), dtype=np.int64)
                for row, tokens in enumerate(seq):
                    ids[row, : len(tokens)] = tokens
                    mask[row, : len(tokens)] = 1
                hidden = self._session.run(["last_hidden_state"], {"input_ids": ids, "attention_mask": mask})[0]
                vectors[batch_idx] = self._pool(hidden, mask)
            return vectors
        except SimgrepError:
            raise
        except Exception as exc:
            raise SimgrepError(f"Failed to generate ONNX embeddings for model '{self._model_name}'.") from exc

    def _pool(self, hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self._pooling == "cls":
            pooled = hidden[:, 0].astype(np.float32)
        else:
            m = mask[:, :, None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        if self._normalize_embeddings:
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
        return pooled.astype(np.float32)
