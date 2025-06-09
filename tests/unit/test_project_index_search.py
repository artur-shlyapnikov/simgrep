import os
from pathlib import Path
from typing import Callable
from unittest.mock import patch

from typer.testing import CliRunner

from simgrep.config import load_or_create_global_config
from simgrep.main import app
from simgrep.metadata_db import connect_global_db, get_project_config

runner = CliRunner()


def _mock_expand(base: Path) -> Callable[[str], str]:
    def _inner(path: str) -> str:
        if path == "~" or path.startswith("~/"):
            return path.replace("~", str(base), 1)
        return os.path.expanduser(path)

    return _inner


def test_project_index_and_search(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "file.txt").write_text("searchterm here")

    with patch("os.path.expanduser", side_effect=_mock_expand(home)):
        result = runner.invoke(app, ["project", "create", "projA"])
        assert result.exit_code == 0

        result = runner.invoke(app, ["project", "add-path", str(data_dir), "--project", "projA"])
        assert result.exit_code == 0

        result = runner.invoke(
            app,
            ["index", "--project", "projA", "--rebuild"],
            input="y\n",
        )
        assert result.exit_code == 0

        cfg = load_or_create_global_config()
        global_db_path = cfg.db_directory / "global_metadata.duckdb"
        conn = connect_global_db(global_db_path)
        try:
            proj_cfg = get_project_config(conn, "projA")
            assert proj_cfg is not None
            assert proj_cfg.db_path.exists()
            assert proj_cfg.usearch_index_path.exists()
        finally:
            conn.close()

        result = runner.invoke(app, ["search", "searchterm", "--project", "projA"])
        assert result.exit_code == 0
        assert "Score:" in result.stdout
