import os
from pathlib import Path
from typing import Callable
from unittest.mock import patch

from typer.testing import CliRunner

from simgrep.config import load_global_config
from simgrep.main import app
from simgrep.metadata_db import connect_global_db, get_project_config

runner = CliRunner()


def _mock_expand(base: Path) -> Callable[[str], str]:
    def _inner(path: str) -> str:
        if path == "~" or path.startswith("~/"):
            return path.replace("~", str(base), 1)
        return os.path.expanduser(path)

    return _inner


def test_project_init_index_and_search_workflow(tmp_path: Path) -> None:
    """
    Tests the core user workflow:
    1. `init --global`
    2. `init` in a local project directory
    3. `index` the project
    4. `search` the project
    """
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    project_dir = tmp_path / "my-test-project"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("The quick brown fox jumps over the lazy dog.")

    with patch("os.path.expanduser", side_effect=_mock_expand(home_dir)):
        # 1. Global init
        result = runner.invoke(app, ["init", "--global"])
        assert result.exit_code == 0, result.stdout

        # 2. Local init
        # Change to project dir to run local init
        original_cwd = os.getcwd()
        try:
            os.chdir(project_dir)
            local_init_result = runner.invoke(app, ["init"])
            assert local_init_result.exit_code == 0, local_init_result.stdout
            assert "Initialized simgrep project 'my-test-project'" in local_init_result.stdout
            assert (project_dir / ".simgrep").exists()

            # 3. Index the project (explicitly, to avoid cwd flakiness)
            index_result = runner.invoke(app, ["index", "--rebuild", "--project", "my-test-project", "--yes"])
            assert index_result.exit_code == 0, index_result.stdout
            assert "Successfully indexed project 'my-test-project'" in index_result.stdout

            # 4. Search the project (explicitly)
            search_result = runner.invoke(app, ["search", "lazy animal", "--project", "my-test-project"])
            assert search_result.exit_code == 0, search_result.stdout
            assert "Score:" in search_result.stdout
            assert "lazy dog" in search_result.stdout
        finally:
            os.chdir(original_cwd)

        # Verify project was created in global DB
        cfg = load_global_config()
        global_db_path = cfg.db_directory / "global_metadata.duckdb"
        conn = connect_global_db(global_db_path)
        try:
            proj_cfg = get_project_config(conn, "my-test-project")
            assert proj_cfg is not None
            assert proj_cfg.db_path.exists()
            assert proj_cfg.usearch_index_path.exists()
        finally:
            conn.close()
