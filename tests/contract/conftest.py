import pathlib

import numpy as np
import pytest

from simgrep.core.abstractions import VectorIndex


@pytest.fixture
def text_file(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "contract_test.txt"
    p.write_text("Some text for contract testing.")
    return p


@pytest.fixture
def vector_index_with_data(vector_index: VectorIndex) -> VectorIndex:
    ndim = vector_index.ndim
    vecs = np.random.rand(5, ndim).astype(np.float32)
    keys = np.arange(5, dtype=np.int64)
    vector_index.add(keys, vecs)
    return vector_index
