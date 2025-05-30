import datetime # Added for timestamp conversion
import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple # Added Any, Dict

import duckdb

# Assuming models.py is in the same directory or simgrep is installed
try:
    from .exceptions import MetadataDBError
    from .models import ChunkData
except ImportError:
    # This fallback might be needed if running scripts directly from the simgrep folder
    # or if the package structure is not fully resolved in some contexts.
    from simgrep.exceptions import MetadataDBError  # type: ignore
    from simgrep.models import ChunkData  # type: ignore

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
            CREATE SEQUENCE IF NOT EXISTS indexed_files_file_id_seq;
        """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS indexed_files (
                file_id BIGINT PRIMARY KEY DEFAULT nextval('indexed_files_file_id_seq'),
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
            CREATE SEQUENCE IF NOT EXISTS text_chunks_chunk_id_seq;
        """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS text_chunks (
                chunk_id BIGINT PRIMARY KEY DEFAULT nextval('text_chunks_chunk_id_seq'),
                file_id BIGINT NOT NULL REFERENCES indexed_files(file_id),
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
        logger.debug(
            f"Foreign key constraints are enforced by default in DuckDB for DB at {db_path}"
        )
    except duckdb.Error as e:
        logger.error(
            f"Failed to connect to or initialize persistent DB at {db_path}: {e}"
        )
        raise MetadataDBError(f"Failed to connect/initialize DB at {db_path}") from e

    _create_persistent_tables_if_not_exist(conn)
    return conn


def clear_persistent_project_data(conn: duckdb.DuckDBPyConnection) -> None:
    logger.info("Clearing all data from 'text_chunks' and 'indexed_files' tables for persistent project.")
    try:
        # Using an explicit transaction for atomicity of DDL-like operations and DML
        with conn.cursor() as cursor:
            cursor.execute("BEGIN TRANSACTION;")
            cursor.execute("DELETE FROM text_chunks;")
            logger.debug("Deleted all records from 'text_chunks'.")
            cursor.execute("DELETE FROM indexed_files;")
            logger.debug("Deleted all records from 'indexed_files'.")
            
            # Reset sequences for auto-incrementing primary keys
            sequences = [seq[0] for seq in cursor.execute("SELECT sequence_name FROM duckdb_sequences();").fetchall()]
            if 'text_chunks_chunk_id_seq' in sequences:
                cursor.execute("ALTER SEQUENCE text_chunks_chunk_id_seq RESTART WITH 1;")
                logger.debug("Reset sequence 'text_chunks_chunk_id_seq'.")
            else:
                logger.warning("Sequence 'text_chunks_chunk_id_seq' not found for reset.")

            if 'indexed_files_file_id_seq' in sequences:
                cursor.execute("ALTER SEQUENCE indexed_files_file_id_seq RESTART WITH 1;")
                logger.debug("Reset sequence 'indexed_files_file_id_seq'.")
            else:
                logger.warning("Sequence 'indexed_files_file_id_seq' not found for reset.")
            
            cursor.execute("COMMIT;")
        logger.info("Persistent project data cleared and sequences reset.")
    except duckdb.Error as e:
        logger.error(f"Error clearing persistent project data: {e}")
        # Attempt to rollback if transaction was started and failed
        try:
            conn.execute("ROLLBACK;")
            logger.info("Rolled back transaction after error in clear_persistent_project_data.")
        except duckdb.Error as rb_err:
            logger.error(f"Failed to rollback transaction: {rb_err}")
        raise MetadataDBError("Failed to clear persistent project data") from e


def insert_indexed_file_record(
    conn: duckdb.DuckDBPyConnection,
    file_path: str, # Absolute, resolved path
    content_hash: str,
    file_size_bytes: int,
    last_modified_os_timestamp: float, # From file_path.stat().st_mtime
) -> Optional[int]:
    logger.debug(f"Attempting to insert metadata for file: {file_path}")
    
    last_modified_dt = datetime.datetime.fromtimestamp(last_modified_os_timestamp)

    try:
        result = conn.execute(
            """
            INSERT INTO indexed_files (file_path, content_hash, file_size_bytes, last_modified_os, last_indexed_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            RETURNING file_id;
            """,
            [file_path, content_hash, file_size_bytes, last_modified_dt],
        ).fetchone()
        
        if result and result[0] is not None:
            file_id = int(result[0])
            logger.info(f"Inserted file '{file_path}' into 'indexed_files' with file_id {file_id}.")
            return file_id
        else:
            logger.error(f"Failed to retrieve file_id after inserting file metadata for '{file_path}'.")
            return None
    except duckdb.ConstraintException as e:
        logger.error(f"Constraint violation for file '{file_path}': {e}.")
        raise MetadataDBError(f"Constraint violation inserting file metadata for '{file_path}'") from e
    except duckdb.Error as e:
        logger.error(f"DuckDB error inserting file metadata for '{file_path}': {e}")
        raise MetadataDBError(f"Failed to insert file metadata for '{file_path}'") from e


def batch_insert_text_chunks(conn: duckdb.DuckDBPyConnection, chunk_records: List[Dict[str, Any]]) -> None:
    if not chunk_records:
        logger.debug("No chunk records provided for batch insert into 'text_chunks'.")
        return

    data_to_insert = []
    for record in chunk_records:
        data_to_insert.append((
            record["file_id"],
            record["usearch_label"],
            record["chunk_text_snippet"],
            record["start_char_offset"],
            record["end_char_offset"],
            record["token_count"],
            record.get("embedding_hash"),
        ))
    
    logger.info(f"Batch inserting {len(data_to_insert)} chunk record(s) into persistent 'text_chunks'.")
    try:
        # Using an explicit transaction for batch insert
        with conn.cursor() as cursor:
            cursor.execute("BEGIN TRANSACTION;")
            cursor.executemany(
                """
                INSERT INTO text_chunks (
                    file_id, usearch_label, chunk_text_snippet,
                    start_char_offset, end_char_offset, token_count, embedding_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                data_to_insert,
            )
            cursor.execute("COMMIT;")
        logger.debug(f"Successfully batch inserted {len(data_to_insert)} chunk record(s) into 'text_chunks'.")
    except duckdb.ConstraintException as e:
        logger.error(f"Constraint violation during batch chunk insert: {e}.")
        try:
            conn.execute("ROLLBACK;")
        except duckdb.Error as rb_err:
            logger.error(f"Failed to rollback transaction: {rb_err}")
        raise MetadataDBError("Constraint violation during batch chunk insert") from e
    except duckdb.Error as e:
        logger.error(f"DuckDB error during batch insert into 'text_chunks': {e}")
        try:
            conn.execute("ROLLBACK;")
        except duckdb.Error as rb_err:
            logger.error(f"Failed to rollback transaction: {rb_err}")
        raise MetadataDBError("Failed during batch insert into 'text_chunks'") from e