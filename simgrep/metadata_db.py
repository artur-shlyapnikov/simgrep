import datetime
import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import duckdb

# assuming models.py is in the same directory or simgrep is installed
try:
    from .exceptions import MetadataDBError
    from .models import ChunkData
except ImportError:
    # this fallback might be needed if running scripts directly from the simgrep folder
    # or if the package structure is not fully resolved in some contexts.
    from simgrep.exceptions import MetadataDBError  # type: ignore
    from simgrep.models import ChunkData  # type: ignore

logger = logging.getLogger(__name__)


def create_inmemory_db_connection() -> duckdb.DuckDBPyConnection:
    """Creates and returns an in-memory DuckDB database connection."""
    logger.info("Creating in-memory DuckDB connection.")
    try:
        conn = duckdb.connect(database=":memory:", read_only=False)
        # duckdb enforces foreign keys by default if defined in schema.
        # the pragma foreign_keys = on; is sqlite syntax.
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
        # temp_files: stores information about each unique file processed.
        conn.execute(
            """
            CREATE TABLE temp_files (
                file_id INTEGER PRIMARY KEY,    -- ephemeral id for the file in this run
                file_path TEXT NOT NULL UNIQUE  -- absolute path to the file
            );
        """
        )
        logger.debug("Table 'temp_files' created for ephemeral use.")

        # temp_chunks: stores detailed information about each chunk.
        conn.execute(
            """
            CREATE TABLE temp_chunks (
                chunk_id INTEGER PRIMARY KEY,         -- corresponds to chunkdata.usearch_label
                file_id INTEGER NOT NULL,             -- fk to temp_files.file_id
                text_content TEXT NOT NULL,           -- full text of the chunk
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


def batch_insert_files(conn: duckdb.DuckDBPyConnection, files_metadata: List[Tuple[int, pathlib.Path]]) -> None:
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
        conn.executemany("INSERT INTO temp_files (file_id, file_path) VALUES (?, ?)", data_to_insert)
        logger.debug(f"Successfully inserted {len(data_to_insert)} file(s).")
    except duckdb.Error as e:
        logger.error(f"DuckDB error during batch file insert: {e}")
        raise MetadataDBError("Failed during batch file insert") from e


def batch_insert_chunks(conn: duckdb.DuckDBPyConnection, chunk_data_list: List[ChunkData]) -> None:
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
        sql = (
            "INSERT INTO temp_chunks (chunk_id, file_id, text_content, "
            "start_char_offset, end_char_offset, token_count) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        )
        conn.executemany(sql, data_to_insert)
        logger.debug(f"Successfully inserted {len(data_to_insert)} chunk(s).")
    except duckdb.Error as e:
        logger.error(f"DuckDB error during batch chunk insert: {e}")
        raise MetadataDBError("Failed during batch chunk insert") from e


def retrieve_chunk_for_display(conn: duckdb.DuckDBPyConnection, chunk_id: int) -> Optional[Tuple[str, pathlib.Path, int, int]]:
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
        # not raising MetadataDBError here as it's a query, not a structural/connection issue.
        # caller should handle optional return.
        return None


def retrieve_chunk_details_persistent(
    conn: duckdb.DuckDBPyConnection, usearch_label: int
) -> Optional[Tuple[str, pathlib.Path, int, int]]:
    """
    Retrieves chunk details (snippet, file path, offsets) from persistent tables
    using the usearch_label.
    """
    query = """
        SELECT tc.chunk_text_snippet, f.file_path, tc.start_char_offset, tc.end_char_offset
        FROM text_chunks tc
        JOIN indexed_files f ON tc.file_id = f.file_id
        WHERE tc.usearch_label = ?;
    """
    logger.debug(f"Retrieving persistent chunk details for usearch_label: {usearch_label}")
    try:
        result = conn.execute(query, [usearch_label]).fetchone()
        if result:
            text_snippet, file_path_str, start_offset, end_offset = result
            logger.debug(f"Chunk with usearch_label {usearch_label} found: {file_path_str}")
            return (
                str(text_snippet),
                pathlib.Path(file_path_str),
                int(start_offset),
                int(end_offset),
            )
        logger.debug(f"Chunk with usearch_label {usearch_label} not found in persistent DB.")
        return None
    except duckdb.Error as e:
        logger.error(f"DuckDB error retrieving persistent chunk (label {usearch_label}): {e}")
        # re-raise as MetadataDBError to signal a problem with db interaction
        raise MetadataDBError(f"Failed to retrieve persistent chunk details for label {usearch_label}") from e


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
                chunk_text_snippet VARCHAR NOT NULL, -- store empty string if no snippet
                start_char_offset INTEGER NOT NULL,
                end_char_offset INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                embedding_hash VARCHAR -- nullable
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
        raise MetadataDBError(f"Could not create directory for database at {db_path.parent}") from e

    try:
        conn = duckdb.connect(database=str(db_path), read_only=False)
        logger.info(f"Successfully connected to persistent DB at {db_path}")
        # duckdb enforces foreign keys by default if defined in schema.
        # the pragma foreign_keys = on; is sqlite syntax.
        logger.debug(f"Foreign key constraints are enforced by default in DuckDB for DB at {db_path}")
    except duckdb.Error as e:
        logger.error(f"Failed to connect to or initialize persistent DB at {db_path}: {e}")
        raise MetadataDBError(f"Failed to connect/initialize DB at {db_path}") from e

    _create_persistent_tables_if_not_exist(conn)
    return conn


def clear_persistent_project_data(conn: duckdb.DuckDBPyConnection) -> None:
    logger.info("Clearing all data from 'text_chunks' and 'indexed_files' tables for persistent project.")
    try:
        # using an explicit transaction for atomicity of ddl-like operations and dml
        with conn.cursor() as cursor:
            cursor.execute("BEGIN TRANSACTION;")
            cursor.execute("DELETE FROM text_chunks;")
            logger.debug("Deleted all records from 'text_chunks'.")
            cursor.execute("DELETE FROM indexed_files;")
            logger.debug("Deleted all records from 'indexed_files'.")

            # resetting sequences with alter sequence ... restart is not supported in duckdb 0.10.0
            # for the purpose of wiping data, simply deleting records is sufficient.
            # primary keys will continue from their last value, which is acceptable.
            logger.info("Sequence reset skipped as 'ALTER SEQUENCE ... RESTART' is not supported in this DuckDB version.")

            cursor.execute("COMMIT;")
        logger.info("Persistent project data cleared.")
    except duckdb.Error as e:
        logger.error(f"Error clearing persistent project data: {e}")
        # attempt to rollback if transaction was started and failed
        try:
            conn.execute("ROLLBACK;")
            logger.info("Rolled back transaction after error in clear_persistent_project_data.")
        except duckdb.Error as rb_err:
            logger.error(f"Failed to rollback transaction: {rb_err}")
        raise MetadataDBError("Failed to clear persistent project data") from e


def insert_indexed_file_record(
    conn: duckdb.DuckDBPyConnection,
    file_path: str,  # absolute, resolved path
    content_hash: str,
    file_size_bytes: int,
    last_modified_os_timestamp: float,  # from file_path.stat().st_mtime
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
        data_to_insert.append(
            (
                record["file_id"],
                record["usearch_label"],
                record["chunk_text_snippet"],
                record["start_char_offset"],
                record["end_char_offset"],
                record["token_count"],
                record.get("embedding_hash"),
            )
        )

    logger.info(f"Batch inserting {len(data_to_insert)} chunk record(s) into persistent 'text_chunks'.")
    try:
        # using an explicit transaction for batch insert
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


def get_indexed_file_record_by_path(conn: duckdb.DuckDBPyConnection, file_path: str) -> Optional[Tuple[int, str]]:
    """Return (file_id, content_hash) for a stored file path if it exists."""
    try:
        row = conn.execute(
            "SELECT file_id, content_hash FROM indexed_files WHERE file_path = ?;",
            [file_path],
        ).fetchone()
        if row:
            return int(row[0]), str(row[1])
        return None
    except duckdb.Error as e:
        logger.error(f"DuckDB error fetching record for {file_path}: {e}")
        raise MetadataDBError("Failed to fetch indexed file record") from e


def get_all_indexed_file_records(conn: duckdb.DuckDBPyConnection) -> List[Tuple[int, str, str]]:
    """Retrieve all indexed file records as (file_id, file_path, content_hash)."""
    try:
        rows = conn.execute("SELECT file_id, file_path, content_hash FROM indexed_files;").fetchall()
        return [(int(r[0]), str(r[1]), str(r[2])) for r in rows]
    except duckdb.Error as e:
        logger.error(f"DuckDB error fetching all indexed file records: {e}")
        raise MetadataDBError("Failed to fetch indexed file records") from e


def delete_file_records(conn: duckdb.DuckDBPyConnection, file_id: int) -> List[int]:
    """Delete all records related to a file and return removed usearch labels."""
    try:
        labels_rows = conn.execute(
            "SELECT usearch_label FROM text_chunks WHERE file_id = ?;",
            [file_id],
        ).fetchall()
        labels = [int(r[0]) for r in labels_rows]
        with conn.cursor() as cursor:
            cursor.execute("BEGIN TRANSACTION;")
            cursor.execute("DELETE FROM text_chunks WHERE file_id = ?;", [file_id])
            cursor.execute("DELETE FROM indexed_files WHERE file_id = ?;", [file_id])
            cursor.execute("COMMIT;")
        return labels
    except duckdb.Error as e:
        logger.error(f"DuckDB error deleting records for file_id {file_id}: {e}")
        try:
            conn.execute("ROLLBACK;")
        except duckdb.Error:
            pass
        raise MetadataDBError("Failed to delete file records") from e


def get_index_counts(conn: duckdb.DuckDBPyConnection) -> Tuple[int, int]:
    try:
        file_res = conn.execute("SELECT COUNT(*) FROM indexed_files;").fetchone()
        chunk_res = conn.execute("SELECT COUNT(*) FROM text_chunks;").fetchone()
        files_count = int(file_res[0]) if file_res else 0
        chunks_count = int(chunk_res[0]) if chunk_res else 0
        return files_count, chunks_count
    except duckdb.Error as e:
        logger.error(f"DuckDB error retrieving index counts: {e}")
        raise MetadataDBError("Failed to retrieve index counts") from e
