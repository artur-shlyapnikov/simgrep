import datetime
import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import duckdb

from .core.errors import MetadataDBError
from .core.models import ChunkData
from .metadata_db import connect_persistent_db, create_inmemory_db_connection

logger = logging.getLogger(__name__)

__all__ = ["MetadataStore"]


class MetadataStore:
    """Convenience wrapper around metadata_db operations."""

    def __init__(self, persistent: bool = False, db_path: Optional[pathlib.Path] = None) -> None:
        self.persistent = persistent
        if persistent:
            if db_path is None:
                raise ValueError("db_path must be provided for persistent MetadataStore")
            self.conn = connect_persistent_db(db_path)
        else:
            self.conn = create_inmemory_db_connection()
            self._setup_ephemeral_tables()

    def _setup_ephemeral_tables(self) -> None:
        """Create in-memory tables for ephemeral search."""
        logger.info("Setting up ephemeral tables (temp_files, temp_chunks).")
        try:
            self.conn.execute(
                """
                CREATE TABLE temp_files (
                    file_id INTEGER PRIMARY KEY,
                    file_path TEXT NOT NULL UNIQUE
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE temp_chunks (
                    chunk_id INTEGER PRIMARY KEY,
                    file_id INTEGER NOT NULL,
                    text_content TEXT NOT NULL,
                    start_char_offset INTEGER NOT NULL,
                    end_char_offset INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    FOREIGN KEY (file_id) REFERENCES temp_files(file_id)
                );
                """
            )
        except Exception as e:
            logger.error(f"Error setting up ephemeral tables: {e}")
            raise MetadataDBError("Failed to set up ephemeral tables") from e

    def close(self) -> None:
        self.conn.close()

    # --- ephemeral table helpers ---
    def batch_insert_files(self, files_metadata: List[Tuple[int, pathlib.Path]]) -> None:
        if not files_metadata:
            logger.debug("No file metadata provided for batch insert.")
            return

        # For ephemeral search, paths can theoretically be duplicated if symlinks
        # resolve to the same file but are passed as distinct entries.
        # We ensure uniqueness on the resolved path before inserting, as the
        # table has a UNIQUE constraint on file_path.
        if not self.persistent:
            seen_paths: set[str] = set()
            unique_metadata: list[tuple[int, pathlib.Path]] = []
            for fid, fpath in files_metadata:
                resolved_path_str = str(fpath.resolve())
                if resolved_path_str not in seen_paths:
                    seen_paths.add(resolved_path_str)
                    unique_metadata.append((fid, fpath))
            files_metadata = unique_metadata

        data_to_insert = [(fid, str(fp.resolve())) for fid, fp in files_metadata]
        table_name = "temp_files" if not self.persistent else "indexed_files"
        try:
            self.conn.executemany(
                f"""
                INSERT INTO {table_name} (file_id, file_path)
                VALUES (?, ?)
                ON CONFLICT (file_id) DO NOTHING;
                """,
                data_to_insert,
            )
        except Exception as e:
            logger.error(f"DuckDB error during batch insert into '{table_name}': {e}")
            raise MetadataDBError(f"Failed during batch insert into '{table_name}'") from e

    def batch_insert_chunks(self, chunk_data_list: List[ChunkData]) -> None:
        if not chunk_data_list:
            logger.debug("No chunk data provided for batch insert.")
            return

        data_to_insert = [
            (
                chunk.usearch_label,
                chunk.source_file_id,
                chunk.text,
                chunk.start_char_offset,
                chunk.end_char_offset,
                chunk.token_count,
            )
            for chunk in chunk_data_list
        ]

        logger.info(f"Batch inserting {len(data_to_insert)} chunk(s) into temp_chunks.")
        try:
            sql = "INSERT INTO temp_chunks (chunk_id, file_id, text_content, " "start_char_offset, end_char_offset, token_count) " "VALUES (?, ?, ?, ?, ?, ?)"
            self.conn.executemany(sql, data_to_insert)
        except Exception as e:
            logger.error(f"DuckDB error during batch chunk insert: {e}")
            raise MetadataDBError("Failed during batch chunk insert") from e

    def retrieve_chunk_for_display(self, chunk_id: int) -> Optional[Tuple[str, pathlib.Path, int, int]]:
        query = """
            SELECT tc.text_content, tf.file_path, tc.start_char_offset, tc.end_char_offset
            FROM temp_chunks tc
            JOIN temp_files tf ON tc.file_id = tf.file_id
            WHERE tc.chunk_id = ?;
        """
        logger.debug(f"Retrieving chunk for display with chunk_id: {chunk_id}")
        try:
            result = self.conn.execute(query, [chunk_id]).fetchone()
            if result:
                text_content, file_path_str, start_offset, end_offset = result
                return (
                    str(text_content),
                    pathlib.Path(file_path_str),
                    int(start_offset),
                    int(end_offset),
                )
            return None
        except Exception as e:
            logger.error(f"DuckDB error retrieving chunk {chunk_id}: {e}")
            return None

    # --- persistent table helpers ---
    def retrieve_chunk_details_persistent(self, usearch_label: int) -> Optional[Tuple[str, pathlib.Path, int, int]]:
        query = """
            SELECT tc.chunk_text, f.file_path, tc.start_char_offset, tc.end_char_offset
            FROM text_chunks tc
            JOIN indexed_files f ON tc.file_id = f.file_id
            WHERE tc.usearch_label = ?;
        """
        logger.debug(f"Retrieving persistent chunk details for usearch_label: {usearch_label}")
        try:
            result = self.conn.execute(query, [usearch_label]).fetchone()
            if result:
                text_content, file_path_str, start_offset, end_offset = result
                return (
                    str(text_content),
                    pathlib.Path(file_path_str),
                    int(start_offset),
                    int(end_offset),
                )
            return None
        except Exception as e:
            logger.error(f"DuckDB error retrieving persistent chunk (label {usearch_label}): {e}")
            raise MetadataDBError(f"Failed to retrieve persistent chunk details for label {usearch_label}") from e

    def retrieve_filtered_chunk_details(
        self,
        usearch_labels: List[int],
        file_filter: Optional[List[str]] = None,
        keyword_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not usearch_labels:
            return []

        params: List[Any] = []
        if self.persistent:
            query_parts = [
                "SELECT f.file_path, tc.chunk_text, tc.start_char_offset, tc.end_char_offset, tc.usearch_label",
                "FROM text_chunks AS tc",
                "JOIN indexed_files AS f ON tc.file_id = f.file_id",
            ]
            label_column = "usearch_label"
            chunk_text_column = "chunk_text"
        else:  # Ephemeral
            query_parts = [
                "SELECT f.file_path, tc.text_content AS chunk_text, tc.start_char_offset, tc.end_char_offset, tc.chunk_id AS usearch_label",
                "FROM temp_chunks AS tc",
                "JOIN temp_files AS f ON tc.file_id = f.file_id",
            ]
            label_column = "chunk_id"
            chunk_text_column = "text_content"

        label_placeholders = ", ".join(["?"] * len(usearch_labels))
        query_parts.append(f"WHERE tc.{label_column} IN ({label_placeholders})")
        params.extend(usearch_labels)

        if file_filter:
            file_filter_clauses = ["f.file_path LIKE ?" for _ in file_filter]
            query_parts.append(f"AND ({' OR '.join(file_filter_clauses)})")
            sql_like_patterns = [p.replace("*", "%") for p in file_filter]
            params.extend(sql_like_patterns)

        if keyword_filter:
            query_parts.append(f"AND lower(tc.{chunk_text_column}) LIKE ?")
            params.append(f"%{keyword_filter.lower()}%")

        full_query = "\n".join(query_parts) + ";"
        logger.debug(f"Executing filtered chunk retrieval query: {full_query} with params: {params}")

        try:
            cursor = self.conn.execute(full_query, params)
            if cursor.description is None:
                return []
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            results: List[Dict[str, Any]] = []
            for row in rows:
                record = dict(zip(columns, row))
                if "file_path" in record and isinstance(record["file_path"], str):
                    record["file_path"] = pathlib.Path(record["file_path"])
                results.append(record)

            return results
        except Exception as e:
            logger.error(f"DuckDB error retrieving filtered chunk details: {e}")
            raise MetadataDBError("Failed to retrieve filtered chunk details") from e

    def clear_persistent_project_data(self) -> None:
        logger.info("Clearing all data from 'indexed_files' and cascading to 'text_chunks'.")
        try:
            # With ON DELETE CASCADE, this will also clear text_chunks
            self.conn.execute("DELETE FROM indexed_files;")
        except Exception as e:
            logger.error(f"Error clearing persistent project data: {e}")
            raise MetadataDBError("Failed to clear persistent project data") from e

    def insert_indexed_file_record(
        self,
        file_path: str,
        content_hash: str,
        file_size_bytes: int,
        last_modified_os_timestamp: float,
    ) -> Optional[int]:
        logger.debug(f"Attempting to insert metadata for file: {file_path}")
        last_modified_dt = datetime.datetime.fromtimestamp(last_modified_os_timestamp)
        try:
            result = self.conn.execute(
                """
                INSERT INTO indexed_files (file_path, content_hash, file_size_bytes, last_modified_os)
                VALUES (?, ?, ?, ?)
                RETURNING file_id;
                """,
                [file_path, content_hash, file_size_bytes, last_modified_dt],
            ).fetchone()
            return int(result[0]) if result else None
        except duckdb.ConstraintException:
            logger.warning(f"File already indexed, skipping: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to insert file record for {file_path}: {e}")
            raise MetadataDBError(f"Failed to insert file record for {file_path}") from e

    def batch_insert_text_chunks(self, chunk_records: List[Dict[str, Any]]) -> None:
        if not chunk_records:
            logger.debug("No chunk records provided for batch insert into 'text_chunks'.")
            return

        data_to_insert = []
        for record in chunk_records:
            data_to_insert.append(
                (
                    record["file_id"],
                    record["usearch_label"],
                    record["chunk_text"],
                    record["start_char_offset"],
                    record["end_char_offset"],
                    record["token_count"],
                    record.get("embedding_hash"),
                )
            )

        logger.info(f"Batch inserting {len(data_to_insert)} chunk record(s) into persistent 'text_chunks'.")
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("BEGIN TRANSACTION;")
                cursor.executemany(
                    """
                    INSERT INTO text_chunks (
                        file_id, usearch_label, chunk_text,
                        start_char_offset, end_char_offset, token_count, embedding_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    data_to_insert,
                )
                cursor.execute("COMMIT;")
        except Exception as e:
            logger.error(f"DuckDB error during batch insert into 'text_chunks': {e}")
            try:
                self.conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise MetadataDBError("Failed during batch insert into 'text_chunks'") from e

    def get_all_indexed_file_records(self) -> List[Tuple[int, str, str]]:
        try:
            rows = self.conn.execute("SELECT file_id, file_path, content_hash FROM indexed_files;").fetchall()
            return [(int(r[0]), str(r[1]), str(r[2])) for r in rows]
        except Exception as e:
            logger.error(f"DuckDB error fetching all indexed file records: {e}")
            raise MetadataDBError("Failed to fetch indexed file records") from e

    def delete_file_records(self, file_id: int) -> List[int]:
        try:
            labels_rows = self.conn.execute(
                "SELECT usearch_label FROM text_chunks WHERE file_id = ?;",
                [file_id],
            ).fetchall()
            labels = [int(r[0]) for r in labels_rows]
            with self.conn.cursor() as cursor:
                cursor.execute("DELETE FROM text_chunks WHERE file_id = ?;", [file_id])
                cursor.execute("DELETE FROM indexed_files WHERE file_id = ?;", [file_id])
            return labels
        except Exception as e:
            logger.error(f"DuckDB error deleting records for file_id {file_id}: {e}")
            try:
                self.conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise MetadataDBError("Failed to delete file records") from e

    def get_index_counts(self) -> Tuple[int, int]:
        """For persistent index, gets the count of indexed files and chunks."""
        if not self.persistent:
            logger.warning("get_index_counts called on a non-persistent store, which is not expected.")
            return 0, 0
        try:
            result = self.conn.execute("SELECT (SELECT COUNT(*) FROM indexed_files), (SELECT COUNT(*) FROM text_chunks);").fetchone()
            if result:
                return int(result[0]), int(result[1])
            return 0, 0
        except Exception as e:
            logger.error(f"DuckDB error getting index counts: {e}")
            raise MetadataDBError("Failed to get index counts") from e

    def get_max_usearch_label(self) -> Optional[int]:
        """Retrieves the maximum usearch label from the metadata table."""
        if not self.persistent:
            return None
        try:
            result = self.conn.execute("SELECT value FROM index_metadata WHERE key = 'max_usearch_label';").fetchone()
            return int(result[0]) if result else None
        except (duckdb.Error, ValueError):
            logger.warning("Could not retrieve max_usearch_label from DB.")
            return None

    def set_max_usearch_label(self, label: int) -> None:
        """Saves the maximum usearch label to the metadata table."""
        if not self.persistent:
            return
        try:
            self.conn.execute(
                "INSERT INTO index_metadata (key, value) VALUES ('max_usearch_label', ?) " "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;",
                [str(label)],
            )
        except duckdb.Error as e:
            logger.error(f"Failed to save max_usearch_label to DB: {e}")
            raise MetadataDBError("Failed to set max usearch label") from e