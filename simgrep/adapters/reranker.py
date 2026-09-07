"""Cross-encoder reranker adapter: lazy sentence-transformers CrossEncoder."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from simgrep.errors import RerankError

_DEFAULT_BATCH_SIZE = 32


class CrossEncoderReranker:
    """Pointwise (query, document) scorer backed by a CrossEncoder model.

    The model is loaded lazily on the first :meth:`score` call — no model I/O
    or heavy import happens at construction time.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder  # lazy: heavy import

                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                raise RerankError(
                    f"failed to load cross-encoder model {self.model_name!r}: {exc}",
                    hint="check the --rerank-model name and your network connection",
                ) from exc
        return self._model

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        """Score each document against ``query``; float32, one per document."""
        if not documents:
            return np.zeros(0, dtype=np.float32)
        model = self._ensure_model()
        pairs = [(query, doc) for doc in documents]
        try:
            scores = model.predict(pairs, batch_size=_DEFAULT_BATCH_SIZE)
        except Exception as exc:
            raise RerankError(
                f"cross-encoder scoring failed for {self.model_name!r}: {exc}",
                hint="check the --rerank-model name and your network connection",
            ) from exc
        return np.asarray(scores, dtype=np.float32)
