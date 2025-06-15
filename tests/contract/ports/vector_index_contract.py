import numpy as np
import pytest
from simgrep.core.abstractions import VectorIndex


@pytest.mark.contract
class VectorIndexContract:
    def test_add_and_len(self, vector_index: VectorIndex):
        ndim = vector_index.ndim
        vecs = np.random.rand(5, ndim).astype(np.float32)
        keys = np.arange(5, dtype=np.int64)
        vector_index.add(keys, vecs)
        assert len(vector_index) == 5

    def test_search_returns_results(self, vector_index_with_data: VectorIndex):
        ndim = vector_index_with_data.ndim
        query = np.random.rand(1, ndim).astype(np.float32)
        results = list(vector_index_with_data.search(query, k=1))
        assert len(results) <= 1
        if results:
            assert hasattr(results[0], "label")
            assert hasattr(results[0], "score")

    def test_save_and_load(self, vector_index_with_data: VectorIndex, tmp_path):
        path = tmp_path / "index.bin"
        vector_index_with_data.save(path)
        assert path.exists()

        new_index = type(vector_index_with_data)(ndim=vector_index_with_data.ndim)
        new_index.load(path)
        assert len(new_index) == len(vector_index_with_data)

    def test_remove_keys(self, vector_index_with_data: VectorIndex):
        initial_len = len(vector_index_with_data)
        assert initial_len > 1
        keys_to_remove = np.array([0, 1], dtype=np.int64)
        vector_index_with_data.remove(keys_to_remove)
        assert len(vector_index_with_data) == initial_len - 2
