import pytest
from simgrep.core.abstractions import Embedder, Repository, TextExtractor, TokenChunker, VectorIndex
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence
from simgrep.core.models import Chunk, SearchResult, ChunkData


# This hook will parametrize contract tests
def pytest_generate_tests(metafunc):
    if "embedder" in metafunc.fixturenames:
        metafunc.parametrize("embedder", ["hf_embedder"], indirect=True)
    if "text_extractor" in metafunc.fixturenames:
        metafunc.parametrize("text_extractor", ["unstructured_extractor"], indirect=True)
    if "token_chunker" in metafunc.fixturenames:
        metafunc.parametrize("token_chunker", ["hf_chunker"], indirect=True)
    if "vector_index" in metafunc.fixturenames:
        metafunc.parametrize("vector_index", ["usearch_index"], indirect=True)
    if "repository" in metafunc.fixturenames:
        metafunc.parametrize("repository", ["metadata_store"], indirect=True)


# --- Fake Implementations for Application Tests ---


class FakeEmbedder(Embedder):
    @property
    def ndim(self) -> int:
        return 3

    def encode(self, texts: List[str], *, is_query: bool = False) -> np.ndarray:
        return np.zeros((len(texts), self.ndim), dtype=np.float32)


@pytest.fixture(scope="session")
def fake_embedder() -> Embedder:
    return FakeEmbedder()


class FakeTextExtractor(TextExtractor):
    def extract(self, path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError
        return "fake text content"


@pytest.fixture(scope="session")
def fake_text_extractor() -> TextExtractor:
    return FakeTextExtractor()


class FakeTokenChunker(TokenChunker):
    def chunk(self, text: str) -> Sequence[Chunk]:
        if not text.strip():
            return []
        return [Chunk(id=0, file_id=0, text=text, start=0, end=len(text), tokens=1)]


@pytest.fixture(scope="session")
def fake_token_chunker() -> TokenChunker:
    return FakeTokenChunker()


class FakeVectorIndex(VectorIndex):
    def __init__(self, ndim: int):
        self._ndim = ndim
        self.data: Dict[int, np.ndarray] = {}
        self.keys_arr = np.array([], dtype=np.int64)

    @property
    def ndim(self) -> int:
        return self._ndim

    def __len__(self) -> int:
        return len(self.data)

    def add(self, keys: np.ndarray, vecs: np.ndarray) -> None:
        for k, v in zip(keys, vecs):
            self.data[int(k)] = v
        self.keys_arr = np.array(list(self.data.keys()), dtype=np.int64)

    def search(self, vec: np.ndarray, k: int) -> List[SearchResult]:
        if not self.data:
            return []
        key = next(iter(self.data.keys()))
        return [SearchResult(label=int(key), score=0.99)]

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass

    @property
    def keys(self) -> np.ndarray:
        return self.keys_arr

    def remove(self, keys: np.ndarray) -> None:
        for k in keys:
            if int(k) in self.data:
                del self.data[int(k)]
        self.keys_arr = np.array(list(self.data.keys()), dtype=np.int64)


@pytest.fixture
def fake_vector_index_factory():
    return FakeVectorIndex


class FakeRepository(Repository):
    def __init__(self):
        self.files: Dict[int, str] = {}
        self.chunks: Dict[int, Dict[str, Any]] = {}
        self.max_label = -1
        self.file_id_counter = 0

    def close(self) -> None:
        pass

    def batch_insert_files(self, files_metadata: List[Tuple[int, Path]]) -> None:
        pass

    def batch_insert_chunks(self, chunk_data_list: List[ChunkData]) -> None:
        pass

    def retrieve_chunk_for_display(self, chunk_id: int) -> Optional[Tuple[str, Path, int, int]]:
        return None

    def retrieve_chunk_details_persistent(self, usearch_label: int) -> Optional[Tuple[str, Path, int, int]]:
        return None

    def retrieve_filtered_chunk_details(
        self,
        usearch_labels: List[int],
        file_filter: Optional[List[str]] = None,
        keyword_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "file_path": Path("/fake/file.txt"),
                "chunk_text": "fake text",
                "usearch_label": label,
                "start_char_offset": 0,
                "end_char_offset": 9,
            }
            for label in usearch_labels
        ]

    def clear_persistent_project_data(self) -> None:
        self.files = {}
        self.chunks = {}

    def insert_indexed_file_record(
        self,
        file_path: str,
        content_hash: str,
        file_size_bytes: int,
        last_modified_os_timestamp: float,
    ) -> Optional[int]:
        file_id = self.file_id_counter
        self.files[file_id] = file_path
        self.file_id_counter += 1
        return file_id

    def batch_insert_text_chunks(self, chunk_records: List[Dict[str, Any]]) -> None:
        for record in chunk_records:
            self.chunks[record["usearch_label"]] = record

    def get_all_indexed_file_records(self) -> List[Tuple[int, str, str]]:
        return []

    def delete_file_records(self, file_id: int) -> List[int]:
        return []

    def get_index_counts(self) -> Tuple[int, int]:
        return (len(self.files), len(self.chunks))

    def get_max_usearch_label(self) -> Optional[int]:
        return self.max_label

    def set_max_usearch_label(self, label: int) -> None:
        self.max_label = label


@pytest.fixture
def fake_repository() -> Repository:
    return FakeRepository()
