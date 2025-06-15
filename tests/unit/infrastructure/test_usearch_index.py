import os
import pathlib
from unittest.mock import patch

import numpy as np
import pytest
import usearch.index

from simgrep.adapters.usearch_index import USearchIndex
from simgrep.core.errors import VectorStoreError
from simgrep.core.models import SearchResult


@pytest.fixture
def simple_embeddings() -> np.ndarray:
    return np.array(
        [
            [0.1, 0.2, 0.3],
            [0.0, 0.1, 0.0],
            [0.2, 0.2, 0.2],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def simple_labels() -> np.ndarray:
    return np.array([10, 20, 30], dtype=np.int64)


class TestUSearchIndexCore:
    def test_valid_embeddings_and_labels(self, simple_embeddings: np.ndarray, simple_labels: np.ndarray) -> None:
        index = USearchIndex(ndim=simple_embeddings.shape[1])
        index.add(keys=simple_labels, vecs=simple_embeddings)

        assert isinstance(index, USearchIndex)
        assert len(index) == simple_embeddings.shape[0]
        assert index.ndim == simple_embeddings.shape[1]

    def test_mismatched_shapes_raises(self) -> None:
        embeddings = np.random.rand(3, 4).astype(np.float32)
        labels = np.array([1, 2], dtype=np.int64)
        index = USearchIndex(ndim=embeddings.shape[1])
        with pytest.raises(
            ValueError,
            match=r"Number of embeddings \(3\) must match number of labels \(2\)",
        ):
            index.add(keys=labels, vecs=embeddings)

    def test_empty_embeddings_allowed(self) -> None:
        embeddings = np.empty((0, 5), dtype=np.float32)
        labels = np.empty((0,), dtype=np.int64)
        index = USearchIndex(ndim=5)
        index.add(keys=labels, vecs=embeddings)
        assert isinstance(index, USearchIndex)
        assert len(index) == 0
        assert index.ndim == 5

    def test_search_returns_results(self, simple_embeddings: np.ndarray, simple_labels: np.ndarray) -> None:
        index = USearchIndex(ndim=simple_embeddings.shape[1])
        index.add(keys=simple_labels, vecs=simple_embeddings)
        query = simple_embeddings[0]
        results = index.search(vec=query, k=2)

        assert results
        assert isinstance(results[0], SearchResult)
        assert results[0].label == simple_labels[0]
        assert pytest.approx(1.0, abs=1e-5) == results[0].score
        assert len(results) <= 2

    def test_dimension_mismatch_errors(self, simple_embeddings: np.ndarray, simple_labels: np.ndarray) -> None:
        index = USearchIndex(ndim=simple_embeddings.shape[1])
        index.add(keys=simple_labels, vecs=simple_embeddings)
        wrong_dim_query = np.random.rand(simple_embeddings.shape[1] + 1).astype(np.float32)
        with pytest.raises(ValueError, match="does not match index dimension"):
            index.search(vec=wrong_dim_query, k=1)

        batch_query = np.random.rand(2, simple_embeddings.shape[1]).astype(np.float32)
        with pytest.raises(ValueError, match="Expected a single query embedding"):
            index.search(vec=batch_query, k=1)

    def test_search_inmemory_respects_k(self, simple_embeddings: np.ndarray, simple_labels: np.ndarray) -> None:
        index = USearchIndex(ndim=simple_embeddings.shape[1])
        index.add(keys=simple_labels, vecs=simple_embeddings)
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result_one = index.search(vec=query, k=1)
        assert len(result_one) == 1

        result_two = index.search(vec=query, k=2)
        assert len(result_two) == 2

    @pytest.mark.parametrize(
        "metric, vecs, query, expected_scores",
        [
            pytest.param(
                "ip",
                np.array([[1.0, 0.0], [0.5, 0.0]], dtype=np.float32),
                np.array([1.0, 0.0], dtype=np.float32),
                {10: 1.0, 20: 0.5},
                id="ip_metric",
            ),
            pytest.param(
                "l2sq",
                np.array([[1.0, 0.0], [0.5, 0.0]], dtype=np.float32),
                np.array([1.0, 0.0], dtype=np.float32),
                {10: 1.0, 20: 1.0 / (1.0 + 0.25)},
                id="l2_metric",
            ),
        ],
    )
    def test_similarity_calculation_for_metrics_real(
        self,
        metric: str,
        vecs: np.ndarray,
        query: np.ndarray,
        expected_scores: dict[int, float],
    ) -> None:
        index = USearchIndex(ndim=2, metric=metric)
        keys = np.array([10, 20], dtype=np.int64)
        index.add(keys=keys, vecs=vecs)

        results = index.search(query, k=2)

        assert len(results) == 2

        # Check that the scores match the expected calculated similarities
        for r in results:
            assert r.label in expected_scores
            assert r.score == pytest.approx(expected_scores[r.label])

    def test_search_on_empty_index(self) -> None:
        index = USearchIndex(ndim=4)
        assert len(index) == 0
        results = index.search(vec=np.random.rand(4).astype(np.float32), k=1)
        assert results == []

    def test_remove_non_existent_keys(self, simple_embeddings: np.ndarray, simple_labels: np.ndarray) -> None:
        index = USearchIndex(ndim=simple_embeddings.shape[1])
        index.add(keys=simple_labels, vecs=simple_embeddings)
        initial_len = len(index)

        # This should not raise an error
        index.remove(np.array([99, 98], dtype=np.int64))

        assert len(index) == initial_len

    def test_init_raises_vector_store_error(self) -> None:
        """Test that VectorStoreError is raised if usearch.index.Index fails on init."""
        with patch("usearch.index.Index", side_effect=Exception("mock init failure")):
            with pytest.raises(VectorStoreError, match="Failed to initialize USearch index"):
                USearchIndex(ndim=4)

    def test_add_invalid_input_types(self) -> None:
        """Test ValueError checks for input types in the add method."""
        index = USearchIndex(ndim=4)
        vecs_np = np.random.rand(2, 4).astype(np.float32)
        keys_np = np.arange(2, dtype=np.int64)

        with pytest.raises(ValueError, match="Embeddings must be a 2D NumPy array"):
            index.add(keys=keys_np, vecs=[[0.1, 0.2, 0.3, 0.4]])  # type: ignore

        with pytest.raises(ValueError, match="Embeddings must be a 2D NumPy array"):
            index.add(keys=keys_np, vecs=np.random.rand(2).astype(np.float32))

        with pytest.raises(ValueError, match="labels must be a 1D NumPy array"):
            index.add(keys=[1, 2], vecs=vecs_np)  # type: ignore

        with pytest.raises(ValueError, match="labels must be a 1D NumPy array"):
            index.add(keys=np.array([[1], [2]]), vecs=vecs_np)

    def test_search_invalid_k_value(self) -> None:
        """Test that searching with k <= 0 raises ValueError."""
        index = USearchIndex(ndim=4)
        with pytest.raises(ValueError, match="k .* must be a positive integer"):
            index.search(vec=np.random.rand(4).astype(np.float32), k=0)
        with pytest.raises(ValueError, match="k .* must be a positive integer"):
            index.search(vec=np.random.rand(4).astype(np.float32), k=-1)

    def test_similarity_calculation_for_unknown_metric(self) -> None:
        """Test the fallback similarity calculation for an unknown metric."""
        index = USearchIndex(ndim=2, metric="haversine")  # usearch supports this
        keys = np.array([10], dtype=np.int64)
        vecs = np.array([[0.1, 0.2]], dtype=np.float32)
        index.add(keys, vecs)

        # Mock the search result to control the distance
        mock_match = usearch.index.Matches(keys=np.array([10]), distances=np.array([0.5]))
        with patch.object(index._index, "search", return_value=mock_match):
            results = index.search(vecs[0], k=1)
            assert len(results) == 1
            # The fallback logic is score = -distance
            assert results[0].score == -0.5

    def test_remove_db_error(self, simple_embeddings: np.ndarray, simple_labels: np.ndarray) -> None:
        """Test that VectorStoreError is raised if removing keys fails."""
        index = USearchIndex(ndim=simple_embeddings.shape[1])
        index.add(keys=simple_labels, vecs=simple_embeddings)
        with patch.object(index._index, "remove", side_effect=Exception("mock remove failure")):
            with pytest.raises(VectorStoreError, match="Failed to remove keys from USearch index"):
                index.remove(np.array([10], dtype=np.int64))


@pytest.fixture
def persistent_index_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Provides a path for a persistent usearch index file within the temp directory."""
    return tmp_path / "persistent_indexes" / "test_simgrep.usearch"


@pytest.fixture
def sample_index_data() -> tuple[np.ndarray, np.ndarray]:
    """Provides sample keys and vectors for an index."""
    ndim = 10
    keys = np.array([10, 20, 30], dtype=np.int64)
    vectors = np.random.rand(3, ndim).astype(np.float32)
    return keys, vectors


@pytest.fixture
def empty_usearch_index() -> USearchIndex:
    """Creates an empty usearch index."""
    return USearchIndex(ndim=5, metric="ip", dtype="f16")


class TestUSearchIndexPersistence:
    def test_save_and_load_persistent_index(
        self,
        sample_index_data: tuple[np.ndarray, np.ndarray],
        persistent_index_path: pathlib.Path,
    ) -> None:
        """Test saving an index and then loading it back."""
        original_keys, original_vectors = sample_index_data
        original_len = len(original_keys)
        original_ndim = original_vectors.shape[1]

        # Create and populate the index inside the test to avoid cross-process issues
        # with libraries that use internal threading (like usearch) and pytest-xdist.
        sample_usearch_index = USearchIndex(ndim=original_ndim)
        sample_usearch_index.add(keys=original_keys, vecs=original_vectors)

        assert not persistent_index_path.exists()
        assert not persistent_index_path.parent.exists()

        sample_usearch_index.save(persistent_index_path)

        assert persistent_index_path.exists()
        assert persistent_index_path.is_file()
        assert persistent_index_path.parent.exists()

        temp_file_path = persistent_index_path.with_suffix(persistent_index_path.suffix + ".tmp")
        assert not temp_file_path.exists()

        loaded_index = USearchIndex(ndim=original_ndim)
        loaded_index.load(persistent_index_path)

        assert isinstance(loaded_index, USearchIndex)
        assert len(loaded_index) == original_len
        assert loaded_index.ndim == original_ndim

        loaded_keys = np.sort(loaded_index.keys)
        assert np.array_equal(np.sort(original_keys), loaded_keys)

    def test_save_empty_index(
        self,
        empty_usearch_index: USearchIndex,
        persistent_index_path: pathlib.Path,
    ) -> None:
        """Test saving an empty index."""
        empty_usearch_index.save(persistent_index_path)
        assert persistent_index_path.exists()

        loaded_index = USearchIndex(ndim=empty_usearch_index.ndim)
        loaded_index.load(persistent_index_path)
        assert len(loaded_index) == 0
        assert loaded_index.ndim == empty_usearch_index.ndim

    def test_load_persistent_index_non_existent_file(self, persistent_index_path: pathlib.Path) -> None:
        """Test loading a non-existent index file raises FileNotFoundError."""
        assert not persistent_index_path.exists()
        index = USearchIndex(ndim=4)
        with pytest.raises(FileNotFoundError):
            index.load(persistent_index_path)

    def test_save_persistent_index_directory_creation_failure(
        self,
        sample_index_data: tuple[np.ndarray, np.ndarray],
        persistent_index_path: pathlib.Path,
    ) -> None:
        """Test handling of OSError if index directory creation fails."""
        keys, vectors = sample_index_data
        sample_usearch_index = USearchIndex(ndim=vectors.shape[1])
        sample_usearch_index.add(keys=keys, vecs=vectors)

        parent_dir_of_index_parent = persistent_index_path.parent.parent
        parent_dir_of_index_parent.mkdir(parents=True, exist_ok=True)

        path_that_should_be_dir = persistent_index_path.parent
        path_that_should_be_dir.touch()

        with pytest.raises(VectorStoreError):
            sample_usearch_index.save(persistent_index_path)
        assert not persistent_index_path.exists()

    def test_save_persistent_index_atomic_failure_on_replace(
        self,
        sample_index_data: tuple[np.ndarray, np.ndarray],
        persistent_index_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that if os.replace fails, the temp file is cleaned up."""
        keys, vectors = sample_index_data
        sample_usearch_index = USearchIndex(ndim=vectors.shape[1])
        sample_usearch_index.add(keys=keys, vecs=vectors)

        temp_file_path = persistent_index_path.with_suffix(persistent_index_path.suffix + ".tmp")

        def mock_os_replace(src: str, dst: str) -> None:
            raise OSError("Simulated failure during os.replace")

        monkeypatch.setattr(os, "replace", mock_os_replace)

        with pytest.raises(VectorStoreError):
            sample_usearch_index.save(persistent_index_path)

        assert not persistent_index_path.exists()
        assert not temp_file_path.exists()

    def test_load_corrupted_index_file(self, persistent_index_path: pathlib.Path) -> None:
        """Test loading a corrupted/invalid index file raises VectorStoreError."""
        persistent_index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(persistent_index_path, "wb") as f:
            f.write(b"this is not a valid usearch index file content")

        index = USearchIndex(ndim=4)
        with pytest.raises(
            VectorStoreError,
            match=f"Failed to load USearch index from {persistent_index_path}",
        ):
            index.load(persistent_index_path)
