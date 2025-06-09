import os
from pathlib import Path
from unittest.mock import patch

from simgrep.config import initialize_global_config, load_global_config
from simgrep.metadata_db import connect_global_db, get_project_by_name


def test_default_project_exists_in_global_db(tmp_path: Path) -> None:
    user_home = tmp_path / "home"
    user_home.mkdir()

    def mock_expand(path_str: str) -> str:
        if path_str == "~" or path_str.startswith("~/"):
            return path_str.replace("~", str(user_home), 1)
        return os.path.expanduser(path_str)

    with patch("os.path.expanduser", side_effect=mock_expand):
        initialize_global_config()
        cfg = load_global_config()
        global_db_path = cfg.db_directory / "global_metadata.duckdb"
        conn = connect_global_db(global_db_path)
        try:
            project = get_project_by_name(conn, "default")
            assert project is not None
        finally:
            conn.close()
