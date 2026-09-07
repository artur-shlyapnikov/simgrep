from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pytest

from simgrep.models import Chunk, VectorHit

if TYPE_CHECKING:
    from simgrep.corpus import StoredChunk
    from simgrep.models import FileRole


class FakeEmbedder:
    @property
    def ndim(self) -> int:
        return 4

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int | None = None) -> np.ndarray:
        del is_query, batch_size
        vectors = np.zeros((len(texts), self.ndim), dtype=np.float32)
        for i, text in enumerate(texts):
            n = float(len(text) or 1)
            vectors[i] = np.array([n, n % 7, n % 13, 1.0], dtype=np.float32)
        return vectors


class FakeTextExtractor:
    def extract(self, path: Path) -> str:
        sample = path.read_bytes()[:8192]
        if b"\x00" in sample:
            return ""
        return path.read_text(encoding="utf-8")


class FakeTokenChunker:
    def chunk(self, text: str) -> Sequence[Chunk]:
        if not text.strip():
            return []
        return [Chunk(id=-1, file_id=-1, text=text, start=0, end=len(text), tokens=max(1, len(text.split())))]


class FakeVectorIndex:
    def __init__(self, ndim: int = 4) -> None:
        self.ndim = ndim
        self.data: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.data)

    def add(
        self,
        labels: np.ndarray | None = None,
        vectors: np.ndarray | None = None,
        *,
        keys: np.ndarray | None = None,
        vecs: np.ndarray | None = None,
    ) -> None:
        actual_labels = labels if labels is not None else keys
        actual_vectors = vectors if vectors is not None else vecs
        assert actual_labels is not None
        assert actual_vectors is not None
        for label, vector in zip(actual_labels, actual_vectors):
            self.data[int(label)] = vector

    def remove(self, labels: np.ndarray | None = None, *, keys: np.ndarray | None = None) -> None:
        actual = labels if labels is not None else keys
        if actual is None:
            return
        for label in actual:
            self.data.pop(int(label), None)

    def search(self, vector: np.ndarray, k: int) -> list[VectorHit]:
        del vector
        return [VectorHit(label=label, score=0.9) for label in sorted(self.data)[:k]]

    def save(self, path: Path) -> None:
        path.write_text(",".join(str(label) for label in sorted(self.data)), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        self.data = {int(label): np.ones(self.ndim, dtype=np.float32) for label in path.read_text(encoding="utf-8").split(",") if label}

    @property
    def keys(self) -> np.ndarray:
        return np.array(sorted(self.data), dtype=np.int64)

    def vectors(self, keys: np.ndarray | None = None) -> np.ndarray:
        actual_keys = self.keys if keys is None else np.asarray(keys, dtype=np.int64)
        if actual_keys.shape[0] == 0:
            return np.zeros((0, self.ndim), dtype=np.float32)
        rows = [np.asarray(self.data[int(key)], dtype=np.float32) for key in actual_keys]
        return np.stack(rows).astype(np.float32, copy=False)


class FakeReranker:
    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        self.calls.append((query, list(documents)))
        return np.asarray([float(len(d) % 7) + self.offset for d in documents], dtype=np.float32) / 7.0


class FakeRuntime:
    def __init__(self) -> None:
        self.extractor = FakeTextExtractor()
        self.chunker = FakeTokenChunker()
        self.embedder = FakeEmbedder()
        self.reranker = FakeReranker()

    @property
    def query_embedder(self) -> "FakeEmbedder":
        # Delegates so tests that swap .embedder keep query and bulk dims consistent.
        return self.embedder

    def require_bulk(self) -> None:
        """Runtime seam: materialize eager components (no-op for fakes)."""
        return None

    def new_vector_index(self, ndim: int) -> FakeVectorIndex:
        return FakeVectorIndex(ndim)


@pytest.fixture
def fake_runtime() -> FakeRuntime:
    return FakeRuntime()


def _ranking_chunk(
    label: int,
    file_path: "Path | str" = "f.py",
    text: str = "x",
    role: "str | FileRole" = "source",
    language: str = "",
    *,
    start_char: int = 0,
    end_char: int = 1,
    token_count: int = 1,
    line_start: int | None = None,
    line_end: int | None = None,
) -> "StoredChunk":
    """Build a StoredChunk for typed rank_candidates/reader fixtures.

    Unknown role strings degrade to ``FileRole.unknown`` (old dict-row behavior).
    """
    from simgrep.corpus import StoredChunk
    from simgrep.models import FileRole as _FileRole

    if isinstance(role, str):
        try:
            resolved = _FileRole(role)
        except ValueError:
            resolved = _FileRole.unknown
    else:
        resolved = role
    return StoredChunk(
        label=int(label),
        file_id=int(label),
        file_path=Path(file_path),
        text=text,
        start_char=start_char,
        end_char=end_char,
        token_count=token_count,
        line_start=line_start,
        line_end=line_end,
        role=resolved,
        language=language,
    )
