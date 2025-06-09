import numpy as np
import pytest
import usearch.index

from simgrep.models import SearchResult
from simgrep.vector_store import create_inmemory_index, search_inmemory_index

pytest.importorskip("numpy")
pytest.importorskip("usearch.index")


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


class TestCreateInmemoryIndex:
    def test_valid_embeddings_and_labels(self, simple_embeddings: np.ndarray, simple_labels: np.ndarray) -> None:
        index = create_inmemory_index(simple_embeddings, simple_labels)

        assert isinstance(index, usearch.index.Index)
        assert len(index) == simple_embeddings.shape[0]
        assert index.ndim == simple_embeddings.shape[1]

    def test_mismatched_shapes_raises(self) -> None:
        embeddings = np.random.rand(3, 4).astype(np.float32)
        labels = np.array([1, 2], dtype=np.int64)
        with pytest.raises(ValueError, match=r"Number of embeddings \(3\) must match number of labels \(2\)"):
            create_inmemory_index(embeddings, labels)

    def test_empty_embeddings_allowed(self) -> None:
        embeddings = np.empty((0, 5), dtype=np.float32)
        labels = np.empty((0,), dtype=np.int64)
        index = create_inmemory_index(embeddings, labels)
        assert isinstance(index, usearch.index.Index)
        assert len(index) == 0
        assert index.ndim == 5


class TestSearchInmemoryIndex:
    def test_search_returns_results(self, simple_embeddings: np.ndarray, simple_labels: np.ndarray) -> None:
        index = create_inmemory_index(simple_embeddings, simple_labels)
        query = simple_embeddings[0]
        results = search_inmemory_index(index, query, k=2)

        assert results
        assert isinstance(results[0], SearchResult)
        assert results[0].label == simple_labels[0]
        assert pytest.approx(1.0, abs=1e-5) == results[0].score
        assert len(results) <= 2

    def test_dimension_mismatch_errors(self, simple_embeddings: np.ndarray, simple_labels: np.ndarray) -> None:
        index = create_inmemory_index(simple_embeddings, simple_labels)
        wrong_dim_query = np.random.rand(simple_embeddings.shape[1] + 1).astype(np.float32)
        with pytest.raises(ValueError, match="does not match index dimension"):
            search_inmemory_index(index, wrong_dim_query, k=1)

        batch_query = np.random.rand(2, simple_embeddings.shape[1]).astype(np.float32)
        with pytest.raises(ValueError, match="Expected a single query embedding"):
            search_inmemory_index(index, batch_query, k=1)
