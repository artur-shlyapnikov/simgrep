from __future__ import annotations

from pathlib import Path

import pytest

from simgrep.errors import ProjectError
from simgrep.project import load_project_config, remove_indexed_path

SCHEMA_VERSION = 1


def _write_project_toml(root: Path, content: str) -> None:
    (root / ".simgrep").mkdir(parents=True, exist_ok=True)
    (root / ".simgrep" / "project.toml").write_text(content, encoding="utf-8")


def _valid_toml(
    model: str = "ibm-granite/granite-embedding-30m-english",
    chunk_size: int = 128,
    chunk_overlap: int = 32,
) -> str:
    return (
        f"schema_version = {SCHEMA_VERSION}\n"
        'name = "proj"\n'
        f'model = "{model}"\n'
        f"chunk_size = {chunk_size}\n"
        f"chunk_overlap = {chunk_overlap}\n"
        'indexed_paths = ["."]\n'
    )


def test_load_project_config_valid(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    _write_project_toml(root, _valid_toml())
    cfg = load_project_config(root)
    assert cfg.model == "ibm-granite/granite-embedding-30m-english"
    assert cfg.chunk_size == 128
    assert cfg.chunk_overlap == 32
    assert cfg.name == "proj"


@pytest.mark.parametrize(
    "toml",
    [
        _valid_toml(model=""),
        _valid_toml(model="   "),
        _valid_toml(chunk_size=0),
        _valid_toml(chunk_size=-3),
        _valid_toml(chunk_overlap=-1),
        _valid_toml(chunk_size=128, chunk_overlap=128),
        _valid_toml(chunk_size=128, chunk_overlap=500),
    ],
)
def test_load_project_config_invalid_values(tmp_path: Path, toml: str) -> None:
    root = tmp_path / "proj"
    _write_project_toml(root, toml)
    with pytest.raises(ProjectError):
        load_project_config(root)


def test_remove_indexed_path_rejects_removing_last_path(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    _write_project_toml(root, _valid_toml())
    cfg = load_project_config(root)
    assert len(cfg.indexed_paths) == 1
    with pytest.raises(ProjectError):
        remove_indexed_path(cfg, cfg.indexed_paths[0])


def test_remove_indexed_path_keeps_remaining_path(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    content = _valid_toml().replace('indexed_paths = ["."]', 'indexed_paths = [".", "sub"]')
    _write_project_toml(root, content)
    cfg = load_project_config(root)
    assert len(cfg.indexed_paths) == 2
    updated = remove_indexed_path(cfg, root / "sub")
    assert updated.indexed_paths == tuple(p for p in cfg.indexed_paths if p.resolve() != (root / "sub").resolve())
    reloaded = load_project_config(root)
    assert reloaded.indexed_paths == updated.indexed_paths


def test_load_project_config_rejects_empty_indexed_paths(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    content = _valid_toml().replace('indexed_paths = ["."]', "indexed_paths = []")
    _write_project_toml(root, content)
    with pytest.raises(ProjectError):
        load_project_config(root)


def test_load_project_config_missing_key_defaults_to_cwd(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    content = _valid_toml().replace('indexed_paths = ["."]\n', "")
    _write_project_toml(root, content)
    cfg = load_project_config(root)
    assert cfg.indexed_paths == (root.resolve(),)
