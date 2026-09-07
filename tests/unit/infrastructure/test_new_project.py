from __future__ import annotations

from pathlib import Path

import pytest

from simgrep.errors import ProjectError
from simgrep.project import (
    add_indexed_path,
    find_active_project,
    init_project,
    load_project_config,
    project_covers_path,
    remove_indexed_path,
)


def test_init_and_load_project_roundtrip(tmp_path: Path) -> None:
    cfg = init_project(tmp_path, name="demo", yes=True)
    loaded = load_project_config(tmp_path)
    assert loaded.name == "demo"
    assert loaded.metadata_db_path == cfg.metadata_db_path
    assert loaded.vector_index_path == cfg.vector_index_path
    assert loaded.index_lock_path == cfg.index_lock_path


def test_find_active_project_walks_up(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    nested = root / "a" / "b" / "c"
    nested.mkdir(parents=True)
    init_project(root, name="root-proj", yes=True)
    found = find_active_project(nested)
    assert found is not None
    assert found.name == "root-proj"


def test_add_remove_indexed_paths_relative_dedup_and_outside_guard(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    cfg = init_project(root, name="repo", yes=True)
    source = root / "src"
    source.mkdir()
    updated = add_indexed_path(cfg, source)
    updated = add_indexed_path(updated, source)
    assert updated.indexed_paths == (root.resolve(), source.resolve())
    removed = remove_indexed_path(updated, source)
    assert removed.indexed_paths == (root.resolve(),)
    outside = tmp_path / "other"
    outside.mkdir()
    with pytest.raises(ProjectError):
        add_indexed_path(cfg, outside)


def test_find_active_project_from_file_start_and_single_file_cover(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    init_project(root, yes=True)
    indexed_file = root / "only.py"
    indexed_file.write_text("x = 1\n", encoding="utf-8")

    assert find_active_project(indexed_file) is not None

    from simgrep.models import SCHEMA_VERSION, ProjectConfig

    file_scoped = ProjectConfig(SCHEMA_VERSION, "repo", root, (indexed_file.resolve(),), "fake", 128, 20)
    assert project_covers_path(file_scoped, indexed_file)
    assert not project_covers_path(file_scoped, root / "other.py")


def test_project_error_paths_for_bad_root_and_missing_config(tmp_path: Path) -> None:
    with pytest.raises(ProjectError, match="existing directory"):
        init_project(tmp_path / "does-not-exist")
    with pytest.raises(ProjectError, match="not found"):
        load_project_config(tmp_path / "empty")
