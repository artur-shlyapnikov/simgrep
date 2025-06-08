import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from hypothesis.extra import numpy as hynp

pytest.importorskip("numpy")
pytest.importorskip("usearch.index")

from simgrep.vector_store import create_inmemory_index, search_inmemory_index


def _embedding_label_strategy():
    @st.composite
    def _inner(draw):
        n = draw(st.integers(min_value=1, max_value=5))
        d = draw(st.integers(min_value=1, max_value=16))
        embeddings = draw(
            hynp.arrays(
                dtype=np.float32,
                shape=(n, d),
                elements=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
            )
        )
        labels_list = draw(st.lists(st.integers(min_value=0, max_value=1000), min_size=n, max_size=n, unique=True))
        labels = np.array(labels_list, dtype=np.int64)
        idx = draw(st.integers(min_value=0, max_value=n - 1))
        query = embeddings[idx]
        return embeddings, labels, query

    return _inner()


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_embedding_label_strategy())
def test_search_results_subset_and_len(data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    embeddings, labels, query = data

    index = create_inmemory_index(embeddings, labels)
    results = search_inmemory_index(index, query, k=5)

    result_keys = {key for key, _ in results}

    assert result_keys.issubset(set(labels.tolist()))
    assert len(results) <= 5
