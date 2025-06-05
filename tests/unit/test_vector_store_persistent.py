import os  # for os.replace mock
import pathlib

import numpy as np
import pytest
import usearch.index

pytest.importorskip("numpy")
pytest.importorskip("usearch.index")

from simgrep.vector_store import (  # noqa: E402
    VectorStoreError,
    load_persistent_index,
    save_persistent_index,
)


@pytest.fixture
def persistent_index_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """provides a path for a persistent usearch index file within the temp directory."""
    return tmp_path / "persistent_indexes" / "test_simgrep.usearch"


@pytest.fixture(scope="module")
def sample_usearch_index() -> usearch.index.Index:
    """creates a sample, non-empty usearch index for testing save/load."""
    ndim = 10
    index = usearch.index.Index(ndim=ndim, metric="cos", dtype="f32")
    keys = np.array([10, 20, 30], dtype=np.int64)
    vectors = np.random.rand(3, ndim).astype(np.float32)
    index.add(keys=keys, vectors=vectors)
    return index


@pytest.fixture(scope="module")
def empty_usearch_index() -> usearch.index.Index:
    """creates an empty usearch index."""
    return usearch.index.Index(ndim=5, metric="ip", dtype="f16")


class TestPersistentVectorStore:
    def test_save_and_load_persistent_index(
        self,
        sample_usearch_index: usearch.index.Index,
        persistent_index_path: pathlib.Path,
    ) -> None:
        """test saving an index and then loading it back."""
        assert not persistent_index_path.exists()
        assert not persistent_index_path.parent.exists()  # to check directory creation

        save_persistent_index(sample_usearch_index, persistent_index_path)

        assert persistent_index_path.exists()
        assert persistent_index_path.is_file()
        assert persistent_index_path.parent.exists()  # directory was created

        # check that the temporary file is gone
        temp_file_path = persistent_index_path.with_suffix(persistent_index_path.suffix + ".tmp")
        assert not temp_file_path.exists()

        loaded_index = load_persistent_index(persistent_index_path)
        assert loaded_index is not None
        assert isinstance(loaded_index, usearch.index.Index)
        assert len(loaded_index) == len(sample_usearch_index)
        assert loaded_index.ndim == sample_usearch_index.ndim
        assert str(loaded_index.metric).lower() == str(sample_usearch_index.metric).lower()  # noqa E501
        assert str(loaded_index.dtype).lower() == str(sample_usearch_index.dtype).lower()

        # verify keys are present (optional, but good for confidence)
        # note: usearch `index.keys()` might not be available or might be slow.
        # `index.get_key(internal_id)` or iterating `index` can get keys.
        # for simplicity, length check is often sufficient for basic save/load test.
        original_keys = np.sort(sample_usearch_index.keys)  # get keys from original if api allows
        loaded_keys = np.sort(loaded_index.keys)
        assert np.array_equal(original_keys, loaded_keys)

    def test_save_empty_index(
        self,
        empty_usearch_index: usearch.index.Index,
        persistent_index_path: pathlib.Path,
    ) -> None:
        """test saving an empty index."""
        save_persistent_index(empty_usearch_index, persistent_index_path)
        assert persistent_index_path.exists()

        loaded_index = load_persistent_index(persistent_index_path)
        assert loaded_index is not None
        assert len(loaded_index) == 0
        assert loaded_index.ndim == empty_usearch_index.ndim

    def test_load_persistent_index_non_existent_file(self, persistent_index_path: pathlib.Path) -> None:
        """test loading a non-existent index file returns none."""
        assert not persistent_index_path.exists()
        loaded_index = load_persistent_index(persistent_index_path)
        assert loaded_index is None

    def test_save_persistent_index_directory_creation_failure(
        self,
        sample_usearch_index: usearch.index.Index,
        persistent_index_path: pathlib.Path,
    ) -> None:
        """test handling of OSError if index directory creation fails."""
        parent_dir_of_index_parent = persistent_index_path.parent.parent
        parent_dir_of_index_parent.mkdir(parents=True, exist_ok=True)

        path_that_should_be_dir = persistent_index_path.parent  # e.g., .../persistent_indexes/
        path_that_should_be_dir.touch()  # create it as a file

        error_match = f"Could not create directory for USearch index at {str(path_that_should_be_dir)}"
        with pytest.raises(VectorStoreError, match=error_match):
            save_persistent_index(sample_usearch_index, persistent_index_path)

    def test_save_persistent_index_atomic_failure_on_replace(
        self,
        sample_usearch_index: usearch.index.Index,
        persistent_index_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """test that if os.replace fails, the temp file is cleaned up."""
        temp_file_path = persistent_index_path.with_suffix(persistent_index_path.suffix + ".tmp")

        def mock_os_replace(src: str, dst: str) -> None:
            # simulate that temp file was created before os.replace is called
            assert temp_file_path.exists()
            raise OSError("Simulated failure during os.replace")

        monkeypatch.setattr(os, "replace", mock_os_replace)

        # the error message comes from the inner try-except block handling os.replace failure.
        # it should indicate failure to finalize the move to the persistent_index_path.
        expected_error_message = f"Failed to finalize saving index to {str(persistent_index_path)}"
        with pytest.raises(VectorStoreError, match=expected_error_message):
            save_persistent_index(sample_usearch_index, persistent_index_path)

        # final state: original file should not exist, temp file should be cleaned up
        assert not persistent_index_path.exists()
        assert not temp_file_path.exists()

    def test_load_corrupted_index_file(self, persistent_index_path: pathlib.Path) -> None:
        """test loading a corrupted/invalid index file raises VectorStoreError."""
        persistent_index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(persistent_index_path, "wb") as f:
            f.write(b"this is not a valid usearch index file content")

        with pytest.raises(VectorStoreError, match=f"Failed to load index from {persistent_index_path}"):
            load_persistent_index(persistent_index_path)
