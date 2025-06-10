import os
from pathlib import Path
from unittest.mock import patch

from simgrep.config import initialize_global_config, load_global_config
from simgrep.project_manager import ProjectManager


def _mock_expand(base: Path):
    def _inner(path: str) -> str:
        if path == "~" or path.startswith("~/"):
            return path.replace("~", str(base), 1)
        return os.path.expanduser(path)

    return _inner


def test_create_add_get(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()

    with patch("os.path.expanduser", side_effect=_mock_expand(home)):
        initialize_global_config()
        cfg = load_global_config()
        manager = ProjectManager(cfg)

        project_cfg = manager.create_project("myproj")
        project_dir = cfg.db_directory / "projects" / "myproj"
        assert project_dir.exists()
        assert project_cfg.db_path == project_dir / "metadata.duckdb"

        sample_path = tmp_path / "data"
        sample_path.mkdir()
        manager.add_path("myproj", sample_path)

        proj_cfg_after = manager.get_config("myproj")
        assert proj_cfg_after is not None
        assert sample_path.resolve() in proj_cfg_after.indexed_paths

        projects = manager.list_projects()
        assert "myproj" in projects
        assert "default" in projects
