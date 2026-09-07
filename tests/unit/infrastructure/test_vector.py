from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import usearch.index

from simgrep.adapters.vector import USearchIndex
from simgrep.errors import SimgrepError
from simgrep.models import VectorHit


def test_vector_add_search_remove_save_load(tmp_path: Path) -> None:
    index = USearchIndex(ndim=3, metric="ip")
    labels = np.array([10, 20], dtype=np.int64)
    vectors = np.array([[1.0, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=np.float32)
    index.add(labels, vectors)
    hits = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=2)
    assert hits and isinstance(hits[0], VectorHit)
    assert hits[0].label == 10
    path = tmp_path / "idx.usearch"
    index.save(path)
    loaded = USearchIndex(ndim=3, metric="ip")
    loaded.load(path)
    loaded_hits = loaded.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)
    assert loaded_hits and loaded_hits[0].label == 10
    loaded.remove(np.array([10], dtype=np.int64))
    assert len(loaded) == 1


def test_add_rejects_wrong_vector_dimensionality() -> None:
    index = USearchIndex(ndim=3)
    labels = np.array([1], dtype=np.int64)
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="Embedding dimensionality mismatch"):
        index.add(labels, vectors)


def test_add_rejects_wrong_labels_shape_or_length() -> None:
    index = USearchIndex(ndim=3)
    vectors = np.array([[1.0, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="labels must be 1D"):
        index.add(np.array([[1, 2]], dtype=np.int64), vectors)

    with pytest.raises(ValueError, match="labels must be 1D"):
        index.add(np.array([1], dtype=np.int64), vectors)


@pytest.mark.parametrize("k", [0, -1])
def test_search_rejects_non_positive_k(k: int) -> None:
    index = USearchIndex(ndim=3)

    with pytest.raises(ValueError, match="k must be positive"):
        index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=k)


def test_search_rejects_wrong_query_shape() -> None:
    index = USearchIndex(ndim=3)

    with pytest.raises(ValueError, match="Query embedding shape"):
        index.search(np.array([[1.0, 0.0]], dtype=np.float32), k=1)


def test_search_on_empty_index_returns_empty_list() -> None:
    index = USearchIndex(ndim=3)

    assert index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=3) == []


def test_remove_nonexistent_labels_is_idempotent() -> None:
    index = USearchIndex(ndim=3)
    index.add(
        np.array([10], dtype=np.int64),
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )

    index.remove(np.array([999], dtype=np.int64))
    index.remove(np.array([999], dtype=np.int64))

    assert len(index) == 1
    assert index.keys.tolist() == [10]


def test_save_load_preserves_keys_and_relevance_order(tmp_path: Path) -> None:
    index = USearchIndex(ndim=3, metric="ip")
    labels = np.array([101, 202, 303], dtype=np.int64)
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.95, 0.0, 0.0],
            [0.90, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    index.add(labels, vectors)
    before = index.search(query, k=3)

    path = tmp_path / "index.usearch"
    index.save(path)

    loaded = USearchIndex(ndim=3, metric="ip")
    loaded.load(path)
    after = loaded.search(query, k=3)

    assert [hit.label for hit in before] == [101, 202, 303]
    assert [hit.label for hit in after] == [101, 202, 303]
    assert [hit.label for hit in after] == [hit.label for hit in before]


@pytest.mark.parametrize(
    ("metric", "distance", "expected"),
    [
        ("ip", 0.25, 0.75),
        ("cos", 0.4, 0.6),
        ("l2sq", 3.0, 0.25),
    ],
)
def test_distance_to_similarity_by_metric(metric: str, distance: float, expected: float) -> None:
    index = USearchIndex(ndim=3, metric=metric)

    assert index._distance_to_similarity(distance) == pytest.approx(expected)


def test_save_cleans_up_tmp_file_when_replace_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index = USearchIndex(ndim=3)
    index.add(np.array([1], dtype=np.int64), np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    path = tmp_path / "broken.usearch"
    tmp_file = path.with_suffix(path.suffix + ".tmp")

    def failing_replace(_src: Path, _dst: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("simgrep.adapters.vector.os.replace", failing_replace)

    with pytest.raises(SimgrepError, match="Failed to save USearch index"):
        index.save(path)

    assert not tmp_file.exists()


def test_init_failure_is_wrapped_in_simgrep_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_index(**_kwargs: object) -> None:
        raise RuntimeError("native boom")

    # usearch is imported lazily via _usearch(); patch that seam, not a module attribute.
    monkeypatch.setattr("simgrep.adapters.vector._usearch", lambda: type("_M", (), {"Index": staticmethod(broken_index)}))

    with pytest.raises(SimgrepError, match="Failed to initialize USearch index"):
        USearchIndex(ndim=3)


def test_empty_index_has_no_keys_and_empty_add_is_a_noop() -> None:
    index = USearchIndex(ndim=3)

    assert index.keys.tolist() == []
    assert index.keys.dtype == np.int64

    index.add(np.array([], dtype=np.int64), np.empty((0, 3), dtype=np.float32))

    assert len(index) == 0


def test_native_failures_are_wrapped_in_simgrep_error(monkeypatch: pytest.MonkeyPatch) -> None:
    index = USearchIndex(ndim=3)
    index.add(np.array([1], dtype=np.int64), np.array([[1.0, 0.0, 0.0]], dtype=np.float32))

    def explode(**_kwargs: object) -> None:
        raise RuntimeError("native failure")

    monkeypatch.setattr(index._index, "add", explode)
    with pytest.raises(SimgrepError, match="Failed to add vectors to USearch index"):
        index.add(np.array([2], dtype=np.int64), np.array([[0.0, 1.0, 0.0]], dtype=np.float32))

    monkeypatch.setattr(index._index, "search", explode)
    with pytest.raises(SimgrepError, match="USearch search operation failed"):
        index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)

    monkeypatch.setattr(index._index, "remove", explode)
    with pytest.raises(SimgrepError, match="Failed to remove vectors from USearch index"):
        index.remove(np.array([1], dtype=np.int64))


def test_load_errors_raise_filenotfound_or_simgrep_error(tmp_path: Path) -> None:
    index = USearchIndex(ndim=3)

    with pytest.raises(FileNotFoundError):
        index.load(tmp_path / "absent.usearch")

    corrupt = tmp_path / "corrupt.usearch"
    corrupt.write_bytes(b"definitely not a usearch file")
    with pytest.raises(SimgrepError, match="Failed to load USearch index"):
        index.load(corrupt)


def test_extract_matches_handles_batch_results_and_unknown_payloads() -> None:
    keys = np.array([[7, 3]], dtype=np.uint64)
    distances = np.array([[0.25, 0.75]], dtype=np.float32)

    batch_keys, batch_distances = USearchIndex._extract_matches(usearch.index.BatchMatches(keys=keys, distances=distances, counts=np.array([2])))
    assert batch_keys.tolist() == [7, 3]
    assert batch_distances.tolist() == [0.25, 0.75]

    empty_keys, empty_distances = USearchIndex._extract_matches(usearch.index.BatchMatches(keys=keys, distances=distances, counts=np.array([0])))
    assert empty_keys.tolist() == []
    assert empty_distances.tolist() == []

    unknown_keys, unknown_distances = USearchIndex._extract_matches(object())
    assert unknown_keys.tolist() == []
    assert unknown_distances.tolist() == []
