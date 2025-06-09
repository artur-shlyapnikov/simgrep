import pathlib
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple

from .metadata_db import (
    batch_insert_chunks,
    batch_insert_files,
    batch_insert_text_chunks,
    clear_persistent_project_data,
    connect_persistent_db,
    create_inmemory_db_connection,
    delete_file_records,
    get_all_indexed_file_records,
    get_index_counts,
    insert_indexed_file_record,
    retrieve_chunk_details_persistent,
    retrieve_chunk_for_display,
    setup_ephemeral_tables,
)
from .models import ChunkData

__all__ = [
    "BaseMetadataStore",
    "EphemeralMetadataStore",
    "PersistentMetadataStore",
]


class BaseMetadataStore(ABC):
    """Base class providing common DB helper methods."""

    def __init__(self, conn) -> None:
        self.conn = conn

    def close(self) -> None:
        self.conn.close()

    # --- ephemeral table helpers ---
    def batch_insert_files(self, files_metadata: List[Tuple[int, pathlib.Path]]) -> None:
        batch_insert_files(self.conn, files_metadata)

    def batch_insert_chunks(self, chunk_data_list: List[ChunkData]) -> None:
        batch_insert_chunks(self.conn, chunk_data_list)

    def retrieve_chunk_for_display(
        self, chunk_id: int
    ) -> Optional[Tuple[str, pathlib.Path, int, int]]:
        return retrieve_chunk_for_display(self.conn, chunk_id)

    # --- persistent table helpers ---
    def retrieve_chunk_details_persistent(
        self, usearch_label: int
    ) -> Optional[Tuple[str, pathlib.Path, int, int]]:
        return retrieve_chunk_details_persistent(self.conn, usearch_label)

    def clear_persistent_project_data(self) -> None:
        clear_persistent_project_data(self.conn)

    def insert_indexed_file_record(
        self,
        file_path: str,
        content_hash: str,
        file_size_bytes: int,
        last_modified_os_timestamp: float,
    ) -> Optional[int]:
        return insert_indexed_file_record(
            self.conn,
            file_path=file_path,
            content_hash=content_hash,
            file_size_bytes=file_size_bytes,
            last_modified_os_timestamp=last_modified_os_timestamp,
        )

    def batch_insert_text_chunks(self, chunk_records: List[Dict[str, Any]]) -> None:
        batch_insert_text_chunks(self.conn, chunk_records)

    def get_all_indexed_file_records(self) -> List[Tuple[int, str, str]]:
        return get_all_indexed_file_records(self.conn)

    def delete_file_records(self, file_id: int) -> List[int]:
        return delete_file_records(self.conn, file_id)

    def get_index_counts(self) -> Tuple[int, int]:
        return get_index_counts(self.conn)


class EphemeralMetadataStore(BaseMetadataStore):
    """In-memory store with temporary tables for ephemeral searches."""

    def __init__(self) -> None:
        conn = create_inmemory_db_connection()
        setup_ephemeral_tables(conn)
        super().__init__(conn)


class PersistentMetadataStore(BaseMetadataStore):
    """Persistent on-disk store for indexing projects."""

    def __init__(self, db_path: pathlib.Path) -> None:
        conn = connect_persistent_db(db_path)
        super().__init__(conn)
