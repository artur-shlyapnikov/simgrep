import numpy as np
import pytest
import usearch.index

from simgrep.vector_store import search_inmemory_index

pytest.importorskip("usearch.index")


@pytest.fixture
def simple_usearch_index() -> usearch.index.Index:
    index = usearch.index.Index(ndim=3, metric="cos", dtype="f32")
    keys = np.array([1, 2, 3], dtype=np.int64)
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    index.add(keys=keys, vectors=vectors)
    return index


def test_search_inmemory_respects_k(simple_usearch_index: usearch.index.Index) -> None:
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    result_one = search_inmemory_index(simple_usearch_index, query, k=1)
    assert len(result_one) == 1

    result_two = search_inmemory_index(simple_usearch_index, query, k=2)
    assert len(result_two) == 2
