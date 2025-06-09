import os
from pathlib import Path
from typing import Callable
from unittest.mock import patch

from typer.testing import CliRunner

from simgrep.config import load_global_config
from simgrep.main import app
from simgrep.metadata_db import connect_global_db, get_all_projects, get_project_by_name

runner = CliRunner()


def _mock_expand(base: Path) -> Callable[[str], str]:
    def _inner(path: str) -> str:
        if path == "~" or path.startswith("~/"):
            return path.replace("~", str(base), 1)
        return os.path.expanduser(path)

    return _inner


def test_project_create_and_list(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    with patch("os.path.expanduser", side_effect=_mock_expand(home)):
        # Global init is required before any project commands can be run
        init_result = runner.invoke(app, ["init", "--global"])
        assert init_result.exit_code == 0

        # create project
        result = runner.invoke(app, ["project", "create", "myproj"])
        assert result.exit_code == 0

        cfg = load_global_config()
        project_dir = cfg.db_directory / "projects" / "myproj"
        assert project_dir.exists()

        db_path = cfg.db_directory / "global_metadata.duckdb"
        conn = connect_global_db(db_path)
        try:
            assert get_project_by_name(conn, "myproj") is not None
            project_names = get_all_projects(conn)
            assert "default" in project_names
            assert "myproj" in project_names
        finally:
            conn.close()

        result = runner.invoke(app, ["project", "list"])
        assert result.exit_code == 0
        assert "myproj" in result.stdout
        assert "default" in result.stdout
