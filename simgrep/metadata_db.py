import pathlib
from typing import List, Optional, Tuple
import logging

import duckdb

# Assuming models.py is in the same directory or simgrep is installed
try:
    from .models import ChunkData
    from .exceptions import MetadataDBError
except ImportError:
    # This fallback might be needed if running scripts directly from the simgrep folder
    # or if the package structure is not fully resolved in some contexts.
    from simgrep.models import ChunkData # type: ignore
    from simgrep.exceptions import MetadataDBError # type: ignore

logger = logging.getLogger(__name__)


def create_inmemory_db_connection() -> duckdb.DuckDBPyConnection:
    """Creates and returns an in-memory DuckDB database connection."""
    logger.info("Creating in-memory DuckDB connection.")
    try:
        conn = duckdb.connect(database=":memory:", read_only=False)
        # DuckDB enforces foreign keys by default if defined in schema.
        # The PRAGMA foreign_keys = ON; is SQLite syntax.
        logger.info("In-memory DuckDB connection established.")
        return conn
    except duckdb.Error as e:
        logger.error(f"Failed to create in-memory DuckDB connection: {e}")
        raise MetadataDBError("Failed to create in-memory DuckDB connection") from e


def setup_ephemeral_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Creates ephemeral tables for files and chunks in the in-memory database.
    These tables are designed for a single ephemeral search operation.
    """
    logger.info("Setting up ephemeral tables (temp_files, temp_chunks).")
    try:
        # temp_files: Stores information about each unique file processed.
        conn.execute(
            """
            CREATE TABLE temp_files (
                file_id INTEGER PRIMARY KEY,    -- Ephemeral ID for the file in this run
                file_path TEXT NOT NULL UNIQUE  -- Absolute path to the file
            );
        """
        )
        logger.debug("Table 'temp_files' created for ephemeral use.")

        # temp_chunks: Stores detailed information about each chunk.
        conn.execute(
            """
            CREATE TABLE temp_chunks (
                chunk_id INTEGER PRIMARY KEY,         -- Corresponds to ChunkData.usearch_label
                file_id INTEGER NOT NULL,             -- FK to temp_files.file_id
                text_content TEXT NOT NULL,           -- Full text of the chunk
                start_char_offset INTEGER NOT NULL,
                end_char_offset INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                FOREIGN KEY (file_id) REFERENCES temp_files(file_id)
            );
        """
        )
        logger.debug("Table 'temp_chunks' created for ephemeral use.")
    except duckdb.Error as e:
        logger.error(f"Error setting up ephemeral tables: {e}")
        raise MetadataDBError("Failed to set up ephemeral tables") from e


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
        logger.debug("No file metadata provided for batch insert.")
        return
    
    data_to_insert = [(fid, str(fp.resolve())) for fid, fp in files_metadata]
    logger.info(f"Batch inserting {len(data_to_insert)} file(s) into temp_files.")
    try:
        conn.executemany(
            "INSERT INTO temp_files (file_id, file_path) VALUES (?, ?)", data_to_insert
        )
        logger.debug(f"Successfully inserted {len(data_to_insert)} file(s).")
    except duckdb.Error as e:
        logger.error(f"DuckDB error during batch file insert: {e}")
        raise MetadataDBError("Failed during batch file insert") from e


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
        conn.executemany(
            "INSERT INTO temp_chunks (chunk_id, file_id, text_content, start_char_offset, end_char_offset, token_count) VALUES (?, ?, ?, ?, ?, ?)",
            data_to_insert,
        )
        logger.debug(f"Successfully inserted {len(data_to_insert)} chunk(s).")
    except duckdb.Error as e:
        logger.error(f"DuckDB error during batch chunk insert: {e}")
        raise MetadataDBError("Failed during batch chunk insert") from e


def retrieve_chunk_for_display(
    conn: duckdb.DuckDBPyConnection, chunk_id: int
) -> Optional[Tuple[str, pathlib.Path, int, int]]:
    """
    Retrieves necessary chunk details for display, given a chunk_id.
    Assumes ephemeral table structure.
    """
    query = """
        SELECT tc.text_content, tf.file_path, tc.start_char_offset, tc.end_char_offset
        FROM temp_chunks tc
        JOIN temp_files tf ON tc.file_id = tf.file_id
        WHERE tc.chunk_id = ?;
    """
    logger.debug(f"Retrieving chunk for display with chunk_id: {chunk_id}")
    try:
        result = conn.execute(query, [chunk_id]).fetchone()
        if result:
            text_content, file_path_str, start_offset, end_offset = result
            logger.debug(f"Chunk {chunk_id} found: {file_path_str}")
            return (
                str(text_content),
                pathlib.Path(file_path_str),
                int(start_offset),
                int(end_offset),
            )
        logger.debug(f"Chunk {chunk_id} not found.")
        return None
    except duckdb.Error as e:
        logger.error(f"DuckDB error retrieving chunk {chunk_id}: {e}")
        # Not raising MetadataDBError here as it's a query, not a structural/connection issue.
        # Caller should handle Optional return.
        return None


def _create_persistent_tables_if_not_exist(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Creates persistent tables (indexed_files, text_chunks) if they don't already exist.
    """
    logger.info("Ensuring persistent tables 'indexed_files' and 'text_chunks' exist.")
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS indexed_files (
                file_id INTEGER PRIMARY KEY, -- Auto-incrementing in DuckDB
                file_path VARCHAR NOT NULL UNIQUE,
                content_hash VARCHAR NOT NULL,
                file_size_bytes BIGINT,
                last_modified_os TIMESTAMP,
                last_indexed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        logger.debug("Table 'indexed_files' ensured.")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS text_chunks (
                chunk_id INTEGER PRIMARY KEY, -- Auto-incrementing
                file_id INTEGER NOT NULL REFERENCES indexed_files(file_id),
                usearch_label BIGINT UNIQUE NOT NULL,
                chunk_text_snippet VARCHAR NOT NULL, -- Store empty string if no snippet
                start_char_offset INTEGER NOT NULL,
                end_char_offset INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                embedding_hash VARCHAR -- Nullable
            );
            """
        )
        logger.debug("Table 'text_chunks' ensured.")
    except duckdb.Error as e:
        logger.error(f"Error creating persistent tables: {e}")
        raise MetadataDBError("Failed to create persistent tables") from e


def connect_persistent_db(db_path: pathlib.Path) -> duckdb.DuckDBPyConnection:
    """
    Connects to a persistent DuckDB database file.
    Creates the directory for the DB if it doesn't exist.
    Ensures necessary tables are created and foreign keys are enabled.
    """
    logger.info(f"Attempting to connect to persistent DuckDB at {db_path}")
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists for DB: {db_path.parent}")
    except OSError as e:
        logger.error(f"Failed to create directory for DB at {db_path.parent}: {e}")
        raise MetadataDBError(
            f"Could not create directory for database at {db_path.parent}"
        ) from e

    try:
        conn = duckdb.connect(database=str(db_path), read_only=False)
        logger.info(f"Successfully connected to persistent DB at {db_path}")
        # DuckDB enforces foreign keys by default if defined in schema.
        # The PRAGMA foreign_keys = ON; is SQLite syntax.
        logger.debug(f"Foreign key constraints are enforced by default in DuckDB for DB at {db_path}")
    except duckdb.Error as e:
        logger.error(
            f"Failed to connect to or initialize persistent DB at {db_path}: {e}"
        )
        raise MetadataDBError(f"Failed to connect/initialize DB at {db_path}") from e

    _create_persistent_tables_if_not_exist(conn)
    return conn