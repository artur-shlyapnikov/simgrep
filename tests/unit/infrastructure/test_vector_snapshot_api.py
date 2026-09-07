from __future__ import annotations

import numpy as np
import pytest

from simgrep.adapters.vector import USearchIndex
from simgrep.errors import SimgrepError


def _seeded_index() -> tuple[USearchIndex, dict[int, np.ndarray]]:
    index = USearchIndex(ndim=4)
    keys = np.array([7, 3, 5], dtype=np.int64)
    vecs = (np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0) / 4.0
    index.add(keys=keys, vecs=vecs)
    expected = {int(key): vecs[i] for i, key in enumerate(keys)}
    return index, expected


def test_add_keys_vectors_round_trip_matches_values() -> None:
    index, expected = _seeded_index()

    assert len(index) == 3
    assert sorted(int(k) for k in index.keys) == [3, 5, 7]

    vectors = index.vectors()
    assert vectors.dtype == np.float32
    assert vectors.shape == (3, 4)
    for row, key in zip(vectors, index.keys):
        assert np.allclose(row, expected[int(key)])


def test_vectors_preserves_explicit_key_order() -> None:
    index, expected = _seeded_index()

    vectors = index.vectors(np.array([5, 3], dtype=np.int64))

    assert vectors.shape == (2, 4)
    assert np.allclose(vectors[0], expected[5])
    assert np.allclose(vectors[1], expected[3])


def test_vectors_empty_index_returns_zero_rows_with_ndim() -> None:
    index = USearchIndex(ndim=6)

    vectors = index.vectors()

    assert vectors.dtype == np.float32
    assert vectors.shape == (0, 6)


def test_vectors_wraps_usearch_failure_in_simgrep_error(monkeypatch: pytest.MonkeyPatch) -> None:
    index = USearchIndex(ndim=4)
    index.add(keys=np.array([1], dtype=np.int64), vecs=np.ones((1, 4), dtype=np.float32))

    def boom(key: int) -> np.ndarray:
        raise RuntimeError(f"usearch get failed for {key}")

    monkeypatch.setattr(index._index, "get", boom)

    with pytest.raises(SimgrepError, match="Failed to fetch vectors"):
        index.vectors()


def test_empty_index_save_load_round_trip(tmp_path: object) -> None:
    from pathlib import Path

    path = Path(str(tmp_path)) / "empty.usearch"
    index = USearchIndex(ndim=4)
    index.save(path)

    loaded = USearchIndex(ndim=4)
    loaded.load(path)

    assert len(loaded) == 0
    vectors = loaded.vectors()
    assert vectors.shape == (0, 4)
