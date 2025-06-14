from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, TypeAlias

import numpy as np

from .models import Chunk, ChunkData, SearchResult

ChunkSeq: TypeAlias = Sequence[Chunk]
ProcessedChunk: TypeAlias = Chunk  # for backward-compatibility
SearchMatches: TypeAlias = Iterable[SearchResult]


class TextExtractor(Protocol):
    def extract(self, path: Path) -> str: ...


class TokenChunker(Protocol):
    def chunk(self, text: str) -> ChunkSeq: ...


class Embedder(Protocol):
    @property
    def ndim(self) -> int: ...

    def encode(self, texts: List[str], *, is_query: bool = False) -> np.ndarray: ...


class VectorIndex(Protocol):
    @property
    def ndim(self) -> int: ...

    def __len__(self) -> int: ...

    def add(self, keys: np.ndarray, vecs: np.ndarray) -> None: ...

    def search(self, vec: np.ndarray, k: int) -> SearchMatches: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...

    @property
    def keys(self) -> np.ndarray: ...

    def remove(self, keys: np.ndarray) -> None: ...


class Repository(Protocol):
    def close(self) -> None: ...

    def batch_insert_files(self, files_metadata: List[Tuple[int, Path]]) -> None: ...

    def batch_insert_chunks(self, chunk_data_list: List[ChunkData]) -> None: ...

    def retrieve_chunk_for_display(self, chunk_id: int) -> Optional[Tuple[str, Path, int, int]]: ...

    def retrieve_chunk_details_persistent(self, usearch_label: int) -> Optional[Tuple[str, Path, int, int]]: ...

    def retrieve_filtered_chunk_details(
        self,
        usearch_labels: List[int],
        file_filter: Optional[List[str]] = None,
        keyword_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]: ...

    def clear_persistent_project_data(self) -> None: ...

    def insert_indexed_file_record(
        self,
        file_path: str,
        content_hash: str,
        file_size_bytes: int,
        last_modified_os_timestamp: float,
    ) -> Optional[int]: ...

    def batch_insert_text_chunks(self, chunk_records: List[Dict[str, Any]]) -> None: ...

    def get_all_indexed_file_records(self) -> List[Tuple[int, str, str]]: ...

    def delete_file_records(self, file_id: int) -> List[int]: ...

    def get_index_counts(self) -> Tuple[int, int]: ...
