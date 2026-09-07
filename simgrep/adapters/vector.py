from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from simgrep.errors import SimgrepError
from simgrep.models import VectorHit

_TORCH_PENDING = False
_USEARCH_MODULE: Any = None


def mark_torch_pending() -> None:
    """Record that a lazy bulk embedder may still import torch in-process."""
    global _TORCH_PENDING
    _TORCH_PENDING = True


def mark_torch_loaded() -> None:
    global _TORCH_PENDING
    _TORCH_PENDING = False


def _usearch() -> Any:
    """Import usearch on first use, after torch when torch is or will be live.

    usearch and torch each bundle an OpenMP runtime; whichever loads first
    wins the process, and torch parallel kernels segfault in libomp barriers
    when usearch's runtime loaded first. Query-only flows never import torch
    (ONNX query embedder), so the guard only forces the torch-first order
    when a lazy bulk embedder is still pending.
    """
    global _USEARCH_MODULE
    if _USEARCH_MODULE is None:
        if "torch" not in sys.modules and _TORCH_PENDING:
            import torch  # noqa: F401
        import usearch.index

        _USEARCH_MODULE = usearch.index
    return _USEARCH_MODULE


class USearchIndex:
    def __init__(self, ndim: int, metric: str = "ip", dtype: str = "f32") -> None:
        self.metric = metric
        try:
            self._index = _usearch().Index(ndim=ndim, metric=metric, dtype=dtype)
        except Exception as exc:
            raise SimgrepError("Failed to initialize USearch index") from exc

    @property
    def ndim(self) -> int:
        return int(self._index.ndim)

    def __len__(self) -> int:
        return len(self._index)

    @property
    def keys(self) -> np.ndarray:
        if len(self._index) == 0:
            return np.array([], dtype=np.int64)
        return np.array([self._index.keys[i] for i in range(len(self._index))], dtype=np.int64)

    def vectors(self, keys: np.ndarray | None = None) -> np.ndarray:
        actual_keys = self.keys if keys is None else np.asarray(keys, dtype=np.int64)
        if actual_keys.shape[0] == 0:
            return np.zeros((0, self.ndim), dtype=np.float32)
        try:
            rows = [np.asarray(self._index.get(int(key)), dtype=np.float32) for key in actual_keys]
        except Exception as exc:
            raise SimgrepError("Failed to fetch vectors from USearch index") from exc
        return np.stack(rows).astype(np.float32, copy=False)

    def add(
        self,
        labels: np.ndarray | None = None,
        vectors: np.ndarray | None = None,
        *,
        keys: np.ndarray | None = None,
        vecs: np.ndarray | None = None,
    ) -> None:
        actual_keys = keys if keys is not None else labels
        actual_vecs = vecs if vecs is not None else vectors
        if actual_keys is None or actual_vecs is None:
            raise ValueError("labels/vectors required")
        if actual_vecs.ndim != 2 or actual_vecs.shape[1] != self.ndim:
            raise ValueError("Embedding dimensionality mismatch")
        if actual_keys.ndim != 1 or actual_keys.shape[0] != actual_vecs.shape[0]:
            raise ValueError("labels must be 1D with one entry per vector")
        if actual_keys.shape[0] == 0:
            return
        try:
            self._index.add(keys=actual_keys.astype(np.int64), vectors=actual_vecs.astype(np.float32), copy=True)
        except Exception as exc:
            raise SimgrepError("Failed to add vectors to USearch index") from exc

    def search(self, vector: np.ndarray, k: int) -> list[VectorHit]:
        if k <= 0:
            raise ValueError("k must be positive")
        query = vector
        if query.ndim == 1:
            query = np.expand_dims(query, axis=0)
        if query.shape != (1, self.ndim):
            raise ValueError(f"Query embedding shape {query.shape} does not match index dimension {self.ndim}")
        if len(self._index) == 0:
            return []
        try:
            result: Any = self._index.search(vectors=query.astype(np.float32), count=k)
        except Exception as exc:
            raise SimgrepError("USearch search operation failed") from exc
        keys, distances = self._extract_matches(result)
        hits: list[VectorHit] = []
        for key, distance in zip(keys, distances):
            hits.append(VectorHit(label=int(key), score=self._distance_to_similarity(float(distance))))
        return hits

    def _distance_to_similarity(self, distance: float) -> float:
        metric = str(self.metric).lower()
        if "cos" in metric or "ip" in metric:
            return 1.0 - distance
        if "l2sq" in metric:
            return 1.0 / (1.0 + distance)
        return -distance

    @staticmethod
    def _extract_matches(result: Any) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(result, _usearch().BatchMatches):
            count = int(result.counts[0]) if result.counts is not None and len(result.counts) else 0
            if count <= 0:
                return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
            return result.keys[0][:count], result.distances[0][:count]
        if isinstance(result, _usearch().Matches):
            return result.keys, result.distances
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    def remove(self, labels: np.ndarray | None = None, *, keys: np.ndarray | None = None) -> None:
        actual = keys if keys is not None else labels
        if actual is None or actual.size == 0:
            return
        try:
            self._index.remove(keys=actual.astype(np.int64))
        except Exception as exc:
            raise SimgrepError("Failed to remove vectors from USearch index") from exc

    def save(self, path: Path) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._index.save(str(temp_path))
            os.replace(temp_path, path)
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            raise SimgrepError(f"Failed to save USearch index to {path}") from exc

    def load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            self._index.load(str(path))
        except Exception as exc:
            raise SimgrepError(f"Failed to load USearch index from {path}") from exc
