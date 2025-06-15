from pathlib import Path
from typing import List, Optional

import duckdb

from .config import SimgrepConfig, load_global_config
from .core.errors import MetadataDBError
from .core.models import ProjectConfig
from .metadata_db import (
    add_project_path,
    connect_global_db,
    create_project_scaffolding,
    get_all_projects,
    get_project_by_name,
    get_project_config,
)


class ProjectManager:
    """High level helper for managing simgrep projects."""

    def __init__(self, config: Optional[SimgrepConfig] = None) -> None:
        self.config = config or load_global_config()
        self._db_path = self.config.db_directory / "global_metadata.duckdb"

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return connect_global_db(self._db_path)

    def create_project(self, name: str) -> ProjectConfig:
        """Create a new project and return its configuration."""
        conn = self._connect()
        try:
            if get_project_by_name(conn, name) is not None:
                raise MetadataDBError(f"Project '{name}' already exists.")
            return create_project_scaffolding(conn, self.config, name)
        finally:
            conn.close()

    def add_path(self, project_name: str, path: Path | str) -> None:
        """Associate a path with the given project."""
        conn = self._connect()
        try:
            proj = get_project_by_name(conn, project_name)
            if proj is None:
                raise MetadataDBError(f"Project '{project_name}' not found.")
            project_id = proj[0]
            add_project_path(conn, project_id, str(path))
        finally:
            conn.close()

    def get_config(self, project_name: str) -> Optional[ProjectConfig]:
        """Return the ``ProjectConfig`` for ``project_name`` if it exists."""
        conn = self._connect()
        try:
            return get_project_config(conn, project_name)
        finally:
            conn.close()

    def list_projects(self) -> List[str]:
        """Return the list of known project names."""
        conn = self._connect()
        try:
            return get_all_projects(conn)
        finally:
            conn.close()
