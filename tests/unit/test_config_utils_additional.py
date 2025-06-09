from pathlib import Path

import pytest

from simgrep.config import (
    DEFAULT_K_RESULTS,
    SimgrepConfig,
    SimgrepConfigError,
    _create_default_project,
    _dumps_toml,
    _serialize_paths,
    add_project_to_config,
    save_config,
)
from simgrep.models import OutputMode, ProjectConfig


def test_serialize_paths_simple(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    result = _serialize_paths(path)
    assert result == str(path)


def test_serialize_paths_list(tmp_path: Path) -> None:
    paths = [tmp_path / "a.txt", tmp_path / "b.txt"]
    serialized = _serialize_paths(paths)
    assert serialized == [str(p) for p in paths]


def test_serialize_paths_nested(tmp_path: Path) -> None:
    data = {"inner": {"p": tmp_path / "c.txt"}}
    serialized = _serialize_paths(data)
    assert serialized == {"inner": {"p": str(tmp_path / "c.txt")}}


def test_dumps_toml_basic() -> None:
    data = {"foo": "bar", "num": 1}
    toml_str = _dumps_toml(data)
    assert "foo = \"bar\"" in toml_str
    assert "num = 1" in toml_str


def test_dumps_toml_with_project(tmp_path: Path) -> None:
    project = {
        "projects": {
            "p": {
                "db_path": str(tmp_path / "db.duckdb"),
                "usearch_index_path": str(tmp_path / "index.usearch"),
            }
        }
    }
    toml_str = _dumps_toml(project)
    assert "[projects.p]" in toml_str


def test_create_default_project_paths(tmp_path: Path) -> None:
    cfg = SimgrepConfig(default_project_data_dir=tmp_path)
    proj = _create_default_project(cfg)
    assert proj.db_path == tmp_path / "metadata.duckdb"
    assert proj.usearch_index_path == tmp_path / "index.usearch"


def test_add_project_to_config(tmp_path: Path) -> None:
    cfg = SimgrepConfig(db_directory=tmp_path, config_file=tmp_path / "cfg.toml")
    added = add_project_to_config(cfg, "newproj")
    assert "newproj" in cfg.projects
    assert added.name == "newproj"


def test_add_project_to_config_duplicate(tmp_path: Path) -> None:
    cfg = SimgrepConfig(db_directory=tmp_path, config_file=tmp_path / "cfg.toml")
    add_project_to_config(cfg, "dup")
    with pytest.raises(SimgrepConfigError):
        add_project_to_config(cfg, "dup")


def test_save_config_writes_file(tmp_path: Path) -> None:
    cfg = SimgrepConfig(config_file=tmp_path / "cfg.toml")
    save_config(cfg)
    assert cfg.config_file.exists()


def test_validate_assignment() -> None:
    cfg = SimgrepConfig()
    cfg.default_chunk_size_tokens = 256
    assert cfg.default_chunk_size_tokens == 256


def test_project_config_defaults() -> None:
    proj = ProjectConfig(
        name="x",
        indexed_paths=[],
        embedding_model="m",
        db_path=Path("a"),
        usearch_index_path=Path("b"),
    )
    assert proj.indexed_paths == []


def test_output_mode_members() -> None:
    assert OutputMode.show.value == "show"
    assert OutputMode.paths.value == "paths"


def test_output_mode_is_str() -> None:
    assert isinstance(OutputMode.show, OutputMode)
    assert isinstance(OutputMode.show.value, str)


def test_default_k_results_constant() -> None:
    assert DEFAULT_K_RESULTS == 5
