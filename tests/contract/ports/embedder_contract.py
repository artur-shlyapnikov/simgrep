import numpy as np
import pytest

from simgrep.core.abstractions import Embedder


@pytest.mark.contract
def test_encode_shape_and_determinism(embedder: Embedder) -> None:
    txt = ["a", "b"]
    arr1 = embedder.encode(txt)
    arr2 = embedder.encode(txt)
    assert arr1.shape == (2, embedder.ndim)
    assert np.allclose(arr1, arr2)
