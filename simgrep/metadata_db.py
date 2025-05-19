import pathlib
from typing import List, Optional, Tuple

import duckdb

# Assuming models.py is in the same directory or simgrep is installed
try:
    from .models import ChunkData
except ImportError:
    # Fallback for potential direct script execution or different project structure
    # This path might be hit if running this file directly for isolated tests in the future.
    from simgrep.models import ChunkData


def create_inmemory_db_connection() -> duckdb.DuckDBPyConnection:
    """Creates and returns an in-memory DuckDB database connection."""
    return duckdb.connect(database=":memory:", read_only=False)


def setup_ephemeral_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Creates ephemeral tables for files and chunks in the in-memory database.
    These tables are designed for a single ephemeral search operation.
    """
    # temp_files: Stores information about each unique file processed.
    # file_id is the ephemeral ID (e.g., index from enumerate(files_to_process)).
    conn.execute(
        """
        CREATE TABLE temp_files (
            file_id INTEGER PRIMARY KEY,    -- Ephemeral ID for the file in this run
            file_path TEXT NOT NULL UNIQUE  -- Absolute path to the file
        );
    """
    )

    # temp_chunks: Stores detailed information about each chunk.
    # chunk_id is the ChunkData.usearch_label, serving as the primary key.
    # file_id links back to temp_files.
    conn.execute(
        """
        CREATE TABLE temp_chunks (
            chunk_id INTEGER PRIMARY KEY,         -- Corresponds to ChunkData.usearch_label
            file_id INTEGER NOT NULL,             -- FK to temp_files.file_id
            text_content TEXT NOT NULL,           -- Full text of the chunk
            start_char_offset INTEGER NOT NULL,
            end_char_offset INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            FOREIGN KEY (file_id) REFERENCES temp_files(file_id) -- Enforce relationship
        );
    """
    )
    # Consider adding indexes if performance becomes an issue, though unlikely for ephemeral.
    # conn.execute("CREATE INDEX idx_chunk_file_id ON temp_chunks(file_id);")


def batch_insert_files(
    conn: duckdb.DuckDBPyConnection, files_metadata: List[Tuple[int, pathlib.Path]]
) -> None:
    """
    Batch inserts file metadata into the temp_files table.
    Args:
        conn: DuckDB connection.
        files_metadata: A list of tuples, each (file_id, file_path).
    """
    if not files_metadata:
        return
    # Convert Path objects to strings for DB insertion.
    data_to_insert = [(fid, str(fp.resolve())) for fid, fp in files_metadata]
    try:
        conn.executemany(
            "INSERT INTO temp_files (file_id, file_path) VALUES (?, ?)", data_to_insert
        )
    except duckdb.Error as e:
        # Handle potential DB errors, e.g., unique constraint violation if logic is flawed
        print(f"DuckDB error during batch file insert: {e}")
        raise  # Re-raise to signal failure


def batch_insert_chunks(
    conn: duckdb.DuckDBPyConnection, chunk_data_list: List[ChunkData]
) -> None:
    """
    Batch inserts chunk data into the temp_chunks table.
    Args:
        conn: DuckDB connection.
        chunk_data_list: A list of ChunkData objects.
    """
    if not chunk_data_list:
        return
    data_to_insert = [
        (
            chunk.usearch_label,  # chunk_id (PK)
            chunk.source_file_id,  # file_id (FK)
            chunk.text,  # text_content
            chunk.start_char_offset,
            chunk.end_char_offset,
            chunk.token_count,
        )
        for chunk in chunk_data_list
    ]
    try:
        conn.executemany(
            "INSERT INTO temp_chunks (chunk_id, file_id, text_content, start_char_offset, end_char_offset, token_count) VALUES (?, ?, ?, ?, ?, ?)",
            data_to_insert,
        )
    except duckdb.Error as e:
        print(f"DuckDB error during batch chunk insert: {e}")
        raise


def retrieve_chunk_for_display(
    conn: duckdb.DuckDBPyConnection, chunk_id: int
) -> Optional[
    Tuple[str, pathlib.Path, int, int]
]:  # text, path, start_offset, end_offset
    """
    Retrieves necessary chunk details for display, given a chunk_id.
    """
    query = """
        SELECT tc.text_content, tf.file_path, tc.start_char_offset, tc.end_char_offset
        FROM temp_chunks tc
        JOIN temp_files tf ON tc.file_id = tf.file_id
        WHERE tc.chunk_id = ?;
    """
    try:
        result = conn.execute(query, [chunk_id]).fetchone()
        if result:
            text_content, file_path_str, start_offset, end_offset = result
            return (
                str(text_content),
                pathlib.Path(file_path_str),
                int(start_offset),
                int(end_offset),
            )
        return None
    except duckdb.Error as e:
        print(f"DuckDB error retrieving chunk {chunk_id}: {e}")
        return None
