import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import usearch.index

from simgrep.core.abstractions import VectorIndex
from simgrep.core.errors import VectorStoreError
from simgrep.core.models import SearchResult

logger = logging.getLogger(__name__)


class USearchIndex(VectorIndex):
    """A USearch-based implementation of the VectorIndex protocol."""

    def __init__(
        self,
        ndim: int,
        metric: str = "cos",
        dtype: str = "f32",
    ):
        try:
            self._index = usearch.index.Index(ndim=ndim, metric=metric, dtype=dtype)
        except Exception as e:
            raise VectorStoreError("Failed to initialize USearch index") from e

    @property
    def ndim(self) -> int:
        return self._index.ndim

    def __len__(self) -> int:
        return len(self._index)

    def add(self, keys: np.ndarray, vecs: np.ndarray) -> None:
        if not isinstance(vecs, np.ndarray) or vecs.ndim != 2:
            raise ValueError("Embeddings must be a 2D NumPy array.")
        if not isinstance(keys, np.ndarray) or keys.ndim != 1:
            raise ValueError("labels must be a 1D NumPy array.")
        if vecs.shape[0] == 0:
            return
        if vecs.shape[0] != keys.shape[0]:
            raise ValueError(f"Number of embeddings ({vecs.shape[0]}) must match number of labels ({keys.shape[0]}).")

        processed_labels = keys
        if keys.size > 0 and keys.dtype != np.int64:
            processed_labels = keys.astype(np.int64)

        try:
            self._index.add(keys=processed_labels, vectors=vecs, copy=True)
        except Exception as e:
            raise VectorStoreError("Failed to add vectors to USearch index") from e

    def search(self, vec: np.ndarray, k: int) -> list[SearchResult]:
        if not isinstance(vec, np.ndarray):
            raise ValueError("Query embedding must be a NumPy array.")
        if k <= 0:
            raise ValueError("k (number of results) must be a positive integer.")

        if len(self) == 0:
            return []

        processed_query_embedding = vec
        if processed_query_embedding.ndim == 1:
            processed_query_embedding = np.expand_dims(processed_query_embedding, axis=0)

        if processed_query_embedding.shape[0] != 1:
            raise ValueError(
                f"Expected a single query embedding, but got batch of {processed_query_embedding.shape[0]}."
            )
        if processed_query_embedding.shape[1] != self.ndim:
            raise ValueError(
                f"Query embedding dimension ({processed_query_embedding.shape[1]}) does not match index dimension ({self.ndim})."
            )

        try:
            search_result: Union[usearch.index.Matches, usearch.index.BatchMatches] = self._index.search(
                vectors=processed_query_embedding, count=k
            )
        except Exception as e:
            raise VectorStoreError("USearch search operation failed") from e

        results: List[SearchResult] = []
        actual_keys: Optional[np.ndarray] = None
        actual_distances: Optional[np.ndarray] = None
        num_found_for_query: int = 0

        if isinstance(search_result, usearch.index.BatchMatches):
            if search_result.counts is not None and len(search_result.counts) > 0:
                num_found_for_query = search_result.counts[0]
                if num_found_for_query > 0:
                    actual_keys = search_result.keys[0]
                    actual_distances = search_result.distances[0]
        elif isinstance(search_result, usearch.index.Matches):
            if hasattr(search_result, "keys") and search_result.keys is not None:
                num_found_for_query = len(search_result.keys)
                if num_found_for_query > 0:
                    actual_keys = search_result.keys
                    actual_distances = search_result.distances
            else:
                num_found_for_query = 0
                actual_keys = None
                actual_distances = None

        if num_found_for_query > 0 and actual_keys is not None and actual_distances is not None:
            for i in range(num_found_for_query):
                key: int = int(actual_keys[i])
                distance: float = float(actual_distances[i])
                similarity: float
                metric_str = str(self._index.metric).lower()

                if "cos" in metric_str:
                    similarity = 1.0 - distance
                elif "ip" in metric_str:
                    similarity = 1.0 - distance
                elif "l2sq" in metric_str:
                    similarity = 1.0 / (1.0 + distance)
                else:
                    similarity = -distance
                results.append(SearchResult(label=key, score=similarity))

        return results

    def save(self, path: Path) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._index.save(str(temp_path))
            os.replace(temp_path, path)
        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise VectorStoreError(f"Failed to save USearch index to {path}") from e

    def load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Index file not found at {path}")
        try:
            self._index.load(str(path))
        except Exception as e:
            raise VectorStoreError(f"Failed to load USearch index from {path}") from e

    @property
    def keys(self) -> np.ndarray:
        count = len(self._index)
        if count == 0:
            return np.array([], dtype=np.int64)
        # Manual iteration to avoid potential issues with list(view) in some environments like pytest-xdist
        keys_list = [self._index.keys[i] for i in range(count)]
        return np.array(keys_list, dtype=np.int64)

    def remove(self, keys: np.ndarray) -> None:
        try:
            self._index.remove(keys=keys.astype(np.int64))
        except Exception as e:
            raise VectorStoreError("Failed to remove keys from USearch index") from e
