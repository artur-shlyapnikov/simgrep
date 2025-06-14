import os
import pathlib

import numpy as np
import pytest
import usearch.index

from simgrep.adapters.usearch_index import USearchIndex
from simgrep.core.errors import VectorStoreError

pytest.importorskip("numpy")
pytest.importorskip("usearch.index")


@pytest.fixture
def persistent_index_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Provides a path for a persistent usearch index file within the temp directory."""
    return tmp_path / "persistent_indexes" / "test_simgrep.usearch"


@pytest.fixture(scope="module")
def sample_usearch_index() -> usearch.index.Index:
    """Creates a sample, non-empty usearch index for testing save/load."""
    ndim = 10
    index = USearchIndex(ndim=ndim, metric="cos", dtype="f32")
    keys = np.array([10, 20, 30], dtype=np.int64)
    vectors = np.random.rand(3, ndim).astype(np.float32)
    index.add(keys=keys, vecs=vectors)
    return index


@pytest.fixture(scope="module")
def empty_usearch_index() -> usearch.index.Index:
    """Creates an empty usearch index."""
    return USearchIndex(ndim=5, metric="ip", dtype="f16")


class TestPersistentVectorStore:
    def test_save_and_load_persistent_index(
        self,
        sample_usearch_index: USearchIndex,
        persistent_index_path: pathlib.Path,
    ) -> None:
        """Test saving an index and then loading it back."""
        assert not persistent_index_path.exists()
        assert not persistent_index_path.parent.exists()

        sample_usearch_index.save(persistent_index_path)

        assert persistent_index_path.exists()
        assert persistent_index_path.is_file()
        assert persistent_index_path.parent.exists()

        temp_file_path = persistent_index_path.with_suffix(persistent_index_path.suffix + ".tmp")
        assert not temp_file_path.exists()

        loaded_index = USearchIndex(ndim=sample_usearch_index.ndim)
        loaded_index.load(persistent_index_path)

        assert isinstance(loaded_index, USearchIndex)
        assert len(loaded_index) == len(sample_usearch_index)
        assert loaded_index.ndim == sample_usearch_index.ndim

        original_keys = np.sort(sample_usearch_index.keys)
        loaded_keys = np.sort(loaded_index.keys)
        assert np.array_equal(original_keys, loaded_keys)

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
        sample_usearch_index: USearchIndex,
        persistent_index_path: pathlib.Path,
    ) -> None:
        """Test handling of OSError if index directory creation fails."""
        parent_dir_of_index_parent = persistent_index_path.parent.parent
        parent_dir_of_index_parent.mkdir(parents=True, exist_ok=True)

        path_that_should_be_dir = persistent_index_path.parent
        path_that_should_be_dir.touch()

        with pytest.raises(VectorStoreError):
            sample_usearch_index.save(persistent_index_path)
        assert not persistent_index_path.exists()

    def test_save_persistent_index_atomic_failure_on_replace(
        self,
        sample_usearch_index: USearchIndex,
        persistent_index_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that if os.replace fails, the temp file is cleaned up."""
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
        with pytest.raises(VectorStoreError, match=f"Failed to load USearch index from {persistent_index_path}"):
            index.load(persistent_index_path)
