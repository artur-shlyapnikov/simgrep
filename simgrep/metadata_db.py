import logging
import pathlib
from typing import List, Optional, Tuple

import duckdb

# assuming models.py is in the same directory or simgrep is installed
try:
    from .exceptions import MetadataDBError
    from .models import ProjectConfig, SimgrepConfig
except ImportError:
    # this fallback might be needed if running scripts directly from the simgrep folder
    # or if the package structure is not fully resolved in some contexts.
    from simgrep.exceptions import MetadataDBError  # type: ignore
    from simgrep.models import ProjectConfig, SimgrepConfig  # type: ignore

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
                chunk_text TEXT NOT NULL,
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




def connect_global_db(path: pathlib.Path) -> duckdb.DuckDBPyConnection:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise MetadataDBError(f"Could not create directory for global DB at {path.parent}") from e

    try:
        conn = duckdb.connect(database=str(path), read_only=False)
    except duckdb.Error as e:
        raise MetadataDBError(f"Failed to connect to global DB at {path}") from e

    try:
        conn.execute("CREATE SEQUENCE IF NOT EXISTS projects_project_id_seq;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                project_id BIGINT PRIMARY KEY DEFAULT nextval('projects_project_id_seq'),
                project_name VARCHAR UNIQUE NOT NULL,
                db_path VARCHAR NOT NULL,
                usearch_index_path VARCHAR NOT NULL,
                embedding_model_name VARCHAR NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS project_indexed_paths (
                project_id BIGINT REFERENCES projects(project_id),
                path VARCHAR NOT NULL,
                PRIMARY KEY (project_id, path)
            );
            """
        )
    except duckdb.Error as e:
        raise MetadataDBError("Failed to create global DB tables") from e

    return conn


def insert_project(
    conn: duckdb.DuckDBPyConnection,
    project_name: str,
    db_path: str,
    usearch_index_path: str,
    embedding_model_name: str,
) -> int:
    try:
        result = conn.execute(
            """
            INSERT INTO projects (project_name, db_path, usearch_index_path, embedding_model_name)
            VALUES (?, ?, ?, ?) RETURNING project_id;
            """,
            [project_name, db_path, usearch_index_path, embedding_model_name],
        ).fetchone()
        return int(result[0]) if result else -1
    except duckdb.Error as e:
        raise MetadataDBError("Failed to insert project") from e


def get_project_by_name(conn: duckdb.DuckDBPyConnection, project_name: str) -> Optional[Tuple[int, str, str, str, str]]:
    try:
        row = conn.execute(
            "SELECT * FROM projects WHERE project_name = ?;",
            [project_name],
        ).fetchone()
        if row:
            return (
                int(row[0]),
                str(row[1]),
                str(row[2]),
                str(row[3]),
                str(row[4]),
            )
        return None
    except duckdb.Error as e:
        raise MetadataDBError("Failed to fetch project") from e


def add_project_path(conn: duckdb.DuckDBPyConnection, project_id: int, path: str) -> None:
    """Adds an indexed path to a project, ignoring duplicates."""
    try:
        # ON CONFLICT is supported in DuckDB >= 0.8.0. pyproject.toml requires >= 0.10.0
        conn.execute(
            "INSERT INTO project_indexed_paths (project_id, path) VALUES (?, ?) ON CONFLICT DO NOTHING;",
            [project_id, path],
        )
        logger.info(f"Added/ensured path '{path}' is associated with project_id {project_id}.")
    except duckdb.Error as e:
        logger.error(f"DuckDB error adding path '{path}' to project {project_id}: {e}")
        raise MetadataDBError(f"Failed to add path to project {project_id}") from e


def get_all_projects(conn: duckdb.DuckDBPyConnection) -> List[str]:
    """Return a list of all project names in the global metadata DB."""
    try:
        rows = conn.execute("SELECT project_name FROM projects ORDER BY project_name;").fetchall()
        return [str(row[0]) for row in rows]
    except duckdb.Error as e:
        raise MetadataDBError("Failed to fetch projects") from e


def get_project_config(conn: duckdb.DuckDBPyConnection, project_name: str) -> Optional[ProjectConfig]:
    """Retrieve a full ProjectConfig from the database."""
    project_row = get_project_by_name(conn, project_name)
    if not project_row:
        return None

    project_id, name, db_path_str, usearch_index_path_str, embedding_model_name = project_row

    path_rows = conn.execute(
        "SELECT path FROM project_indexed_paths WHERE project_id = ?;",
        [project_id],
    ).fetchall()

    indexed_paths = [pathlib.Path(row[0]) for row in path_rows]

    return ProjectConfig(
        name=name,
        indexed_paths=indexed_paths,
        embedding_model=embedding_model_name,
        db_path=pathlib.Path(db_path_str),
        usearch_index_path=pathlib.Path(usearch_index_path_str),
    )


def create_project_scaffolding(
    global_conn: duckdb.DuckDBPyConnection,
    global_config: SimgrepConfig,
    project_name: str,
) -> ProjectConfig:
    """
    Creates project data directories and inserts the project record into the global DB.
    Returns the new ProjectConfig.
    """
    project_data_root = global_config.db_directory / "projects" / project_name
    project_data_root.mkdir(parents=True, exist_ok=True)

    proj_cfg = ProjectConfig(
        name=project_name,
        indexed_paths=[],
        embedding_model=global_config.default_embedding_model_name,
        db_path=project_data_root / "metadata.duckdb",
        usearch_index_path=project_data_root / "index.usearch",
    )

    insert_project(
        global_conn,
        project_name=proj_cfg.name,
        db_path=str(proj_cfg.db_path),
        usearch_index_path=str(proj_cfg.usearch_index_path),
        embedding_model_name=proj_cfg.embedding_model,
    )
    return proj_cfg
