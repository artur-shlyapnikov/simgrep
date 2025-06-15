import pathlib
from unittest.mock import MagicMock, patch

import duckdb
import pytest

from simgrep.core.errors import MetadataDBError
from simgrep.core.models import SimgrepConfig
from simgrep.metadata_db import (
    _create_persistent_tables_if_not_exist,
    add_project_path,
    connect_global_db,
    connect_persistent_db,
    create_project_scaffolding,
    get_all_projects,
    get_project_by_name,
    get_project_config,
    insert_project,
)


@pytest.fixture
def global_db_conn(tmp_path: pathlib.Path) -> duckdb.DuckDBPyConnection:
    db_path = tmp_path / "global_metadata.duckdb"
    conn = connect_global_db(db_path)
    yield conn
    conn.close()


class TestMetadataDb:
    def test_connect_persistent_db_handles_os_error_on_mkdir(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify MetadataDBError is raised if creating the DB directory fails."""
        mock_mkdir = patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied"))

        with mock_mkdir:
            with pytest.raises(MetadataDBError, match="Could not create directory for database"):
                connect_persistent_db(tmp_path / "nonexistent_dir" / "db.duckdb")

    def test_get_project_by_name_returns_none_for_missing_project(self, global_db_conn: duckdb.DuckDBPyConnection) -> None:
        """Ensure get_project_by_name returns None for a non-existent project."""
        result = get_project_by_name(global_db_conn, "non-existent-project")
        assert result is None

    def test_insert_project_handles_uniqueness_constraint(self, global_db_conn: duckdb.DuckDBPyConnection) -> None:
        """Verify inserting a duplicate project name raises MetadataDBError."""
        insert_project(global_db_conn, "test-project", "/db", "/idx", "model")

        with pytest.raises(MetadataDBError, match="Failed to insert project"):
            insert_project(global_db_conn, "test-project", "/db2", "/idx2", "model2")

    def test_get_project_config_for_missing_project(self, global_db_conn: duckdb.DuckDBPyConnection) -> None:
        """Ensure get_project_config returns None for a non-existent project."""
        result = get_project_config(global_db_conn, "non-existent-project")
        assert result is None

    def test_create_project_scaffolding(self, global_db_conn: duckdb.DuckDBPyConnection, tmp_path: pathlib.Path) -> None:
        """Test project scaffolding creation."""
        config = SimgrepConfig(db_directory=tmp_path)
        project_name = "new-scaffold-project"

        proj_cfg = create_project_scaffolding(global_db_conn, config, project_name)

        assert proj_cfg.name == project_name
        assert proj_cfg.db_path.parent.exists()

        # Verify it's in the global DB
        retrieved = get_project_by_name(global_db_conn, project_name)
        assert retrieved is not None
        assert retrieved[1] == project_name

    def test_add_project_path_handles_db_error(self, global_db_conn: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test error handling in add_project_path."""
        monkeypatch.setattr(duckdb.DuckDBPyConnection, "execute", MagicMock(side_effect=duckdb.Error("mock error")))

        with pytest.raises(MetadataDBError):
            add_project_path(global_db_conn, 1, "/some/path")

    def test_connect_persistent_db_connection_error(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify MetadataDBError is raised if duckdb.connect fails."""
        monkeypatch.setattr(duckdb, "connect", MagicMock(side_effect=duckdb.Error("Connection failed")))
        with pytest.raises(MetadataDBError, match="Failed to connect/initialize DB"):
            connect_persistent_db(tmp_path / "db.duckdb")

    def test_create_tables_if_not_exist_db_error(self, tmp_path: pathlib.Path) -> None:
        """Cover the except duckdb.Error block in _create_persistent_tables_if_not_exist."""
        conn = duckdb.connect(":memory:")
        with patch("duckdb.DuckDBPyConnection.execute", side_effect=duckdb.Error("mock table creation error")):
            with pytest.raises(MetadataDBError, match="Failed to create persistent tables"):
                _create_persistent_tables_if_not_exist(conn)
        conn.close()

    def test_get_all_projects_db_error(self, global_db_conn: duckdb.DuckDBPyConnection) -> None:
        """Cover the except duckdb.Error block in get_all_projects."""
        with patch("duckdb.DuckDBPyConnection.execute", side_effect=duckdb.Error("mock db error")):
            with pytest.raises(MetadataDBError, match="Failed to fetch projects"):
                get_all_projects(global_db_conn)