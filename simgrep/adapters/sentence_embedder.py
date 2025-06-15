from functools import lru_cache
from typing import Any, Dict, List, cast

import numpy as np
from sentence_transformers import SentenceTransformer

from simgrep.core.abstractions import Embedder
from simgrep.core.errors import SimgrepError


@lru_cache(maxsize=None)
def _load_embedding_model(model_name: str) -> SentenceTransformer:
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        raise SimgrepError(f"Failed to load embedding model '{model_name}'. Original error: {e}") from e


class SentenceEmbedder(Embedder):
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = _load_embedding_model(model_name)

        dummy_emb = self.encode(["simgrep"], is_query=False)
        if dummy_emb.ndim != 2 or dummy_emb.shape[0] == 0 or dummy_emb.shape[1] == 0:
            raise SimgrepError(f"Could not determine embedding dimension for model {model_name}")
        self._ndim = int(dummy_emb.shape[1])

    @property
    def ndim(self) -> int:
        return self._ndim

    def encode(self, texts: List[str], *, is_query: bool = False) -> np.ndarray:
        try:
            encode_kwargs: Dict[str, Any] = {"show_progress_bar": False}
            if is_query and "qwen" in self._model_name.lower():
                encode_kwargs["prompt_name"] = "query"

            embeddings = self._model.encode(sentences=texts, **encode_kwargs)
            return cast(np.ndarray, embeddings)
        except Exception as e:
            raise SimgrepError(f"Failed to generate embeddings using model '{self._model_name}'. Original error: {e}") from e
