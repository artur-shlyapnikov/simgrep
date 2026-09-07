from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import tomli_w

from simgrep.config import load_app_config, set_config_value
from simgrep.errors import ConfigError, ProjectError
from simgrep.project import (
    add_indexed_path,
    init_project,
    load_project_config,
    project_covers_path,
    remove_indexed_path,
)


class TestProjectInit:
    def test_init_without_yes_fails_if_project_already_exists(self, tmp_path: Path) -> None:
        init_project(tmp_path, name="demo", yes=True)
        with pytest.raises(ProjectError) as exc_info:
            init_project(tmp_path, name="demo")
        assert "Use --yes" in str(exc_info.value.hint)

    def test_init_with_yes_overwrites_existing_project(self, tmp_path: Path) -> None:
        init_project(tmp_path, name="original", yes=True)
        cfg = init_project(tmp_path, name="overwritten", yes=True)
        assert cfg.name == "overwritten"

    def test_init_without_yes_succeeds_when_no_project(self, tmp_path: Path) -> None:
        cfg = init_project(tmp_path, name="fresh")
        assert cfg.name == "fresh"


class TestProjectConfigValidation:
    def test_invalid_project_toml_gives_project_error_with_parse_hint(self, tmp_path: Path) -> None:
        simgrep_dir = tmp_path / ".simgrep"
        simgrep_dir.mkdir()
        project_file = simgrep_dir / "project.toml"
        project_file.write_text("invalid toml {{{{", encoding="utf-8")
        with pytest.raises(ProjectError) as exc_info:
            load_project_config(tmp_path)
        assert "hint" in str(exc_info.value).lower() or "TOML" in str(exc_info.value)

    def test_unsupported_project_schema_gives_project_error(self, tmp_path: Path) -> None:
        simgrep_dir = tmp_path / ".simgrep"
        simgrep_dir.mkdir()
        project_file = simgrep_dir / "project.toml"
        tomli_w.dump({"schema_version": 999, "name": "test"}, project_file.open("wb"))
        with pytest.raises(ProjectError) as exc_info:
            load_project_config(tmp_path)
        assert "schema" in str(exc_info.value).lower()

    def test_indexed_paths_not_list_gives_project_error(self, tmp_path: Path) -> None:
        simgrep_dir = tmp_path / ".simgrep"
        simgrep_dir.mkdir()
        project_file = simgrep_dir / "project.toml"
        tomli_w.dump({"schema_version": 1, "name": "test", "indexed_paths": "not-a-list"}, project_file.open("wb"))
        with pytest.raises(ProjectError) as exc_info:
            load_project_config(tmp_path)
        assert "list" in str(exc_info.value).lower()

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("schema_version", "abc"),
            ("chunk_size", "abc"),
            ("chunk_overlap", "abc"),
            ("schema_version", [1]),
            ("chunk_size", [1]),
            ("chunk_overlap", [1]),
        ],
    )
    def test_non_integer_numeric_field_gives_project_error_naming_key(self, tmp_path: Path, key: str, value: object) -> None:
        simgrep_dir = tmp_path / ".simgrep"
        simgrep_dir.mkdir()
        project_file = simgrep_dir / "project.toml"
        tomli_w.dump({key: value}, project_file.open("wb"))
        with pytest.raises(ProjectError) as exc_info:
            load_project_config(tmp_path)
        assert key in str(exc_info.value)


class TestIndexedPathsStorage:
    def test_relative_indexed_paths_stored_relative_and_loaded_absolute(self, tmp_path: Path) -> None:
        cfg = init_project(tmp_path, name="test", yes=True)
        src = tmp_path / "src"
        src.mkdir()
        add_indexed_path(cfg, src)
        project_file = tmp_path / ".simgrep" / "project.toml"
        with project_file.open("rb") as f:
            saved = tomllib.load(f)
        assert saved["indexed_paths"][-1] == "src"
        reloaded = load_project_config(tmp_path)
        assert reloaded.indexed_paths[-1] == src.resolve()

    def test_outside_root_path_without_flag_fails(self, tmp_path: Path) -> None:
        cfg = init_project(tmp_path, name="test", yes=True)
        outside = tmp_path.parent / "outside"
        outside.mkdir(exist_ok=True)
        with pytest.raises(ProjectError):
            add_indexed_path(cfg, outside)

    def test_outside_root_path_with_flag_stored_absolute_and_works(self, tmp_path: Path) -> None:
        cfg = init_project(tmp_path, name="test", yes=True)
        outside = tmp_path.parent / "outside"
        outside.mkdir(exist_ok=True)
        updated = add_indexed_path(cfg, outside, allow_outside_root=True)
        assert updated.indexed_paths[-1] == outside.resolve()
        reloaded = load_project_config(tmp_path)
        assert reloaded.indexed_paths[-1] == outside.resolve()


class TestRemoveIndexedPath:
    def test_remove_nonexistent_path_is_idempotent(self, tmp_path: Path) -> None:
        cfg = init_project(tmp_path, name="test", yes=True)
        src = tmp_path / "src"
        src.mkdir()
        updated = add_indexed_path(cfg, src)
        result = remove_indexed_path(updated, tmp_path / "nonexistent")
        assert result.indexed_paths == updated.indexed_paths


class TestProjectCoversPath:
    def test_project_covers_path_for_file_indexed(self, tmp_path: Path) -> None:
        cfg = init_project(tmp_path, name="test", yes=True)
        src = tmp_path / "src"
        src.mkdir()
        file_path = src / "file.txt"
        file_path.touch()
        updated = add_indexed_path(cfg, file_path)
        assert project_covers_path(updated, file_path) is True
        outside_root = Path("/usr/bin/python3")
        assert project_covers_path(updated, outside_root) is False

    def test_project_covers_path_for_directory_indexed(self, tmp_path: Path) -> None:
        cfg = init_project(tmp_path, name="test", yes=True)
        src = tmp_path / "src"
        src.mkdir()
        (src / "file.txt").touch()
        updated = add_indexed_path(cfg, src)
        assert project_covers_path(updated, src / "file.txt") is True
        outside_root = Path("/usr")
        assert project_covers_path(updated, outside_root) is False


class TestAppConfigValidation:
    def test_invalid_app_config_toml_gives_config_error(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text("invalid toml {{{{", encoding="utf-8")
        with pytest.raises(ConfigError) as exc_info:
            load_app_config(config_file=config_file)
        assert "hint" in str(exc_info.value).lower() or "TOML" in str(exc_info.value)

    def test_unsupported_app_schema_fallbacks(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        tomli_w.dump({"schema_version": 999}, config_file.open("wb"))
        cfg = load_app_config(config_file=config_file)
        assert cfg.schema_version == 1

    def test_config_set_unknown_key_fails(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        with pytest.raises(ConfigError) as exc_info:
            set_config_value("unknown_key", "value", config_file=config_file)
        assert "unknown" in str(exc_info.value).lower()

    def test_config_set_freshness_invalid_fails(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        with pytest.raises((ConfigError, ValueError)):
            set_config_value("freshness", "invalid_mode", config_file=config_file)

    def test_config_set_file_patterns_saves_tuple(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        cfg = set_config_value("file_patterns", "*.py,*.md", config_file=config_file)
        assert isinstance(cfg.file_patterns, tuple)
        assert "*.py" in cfg.file_patterns
        assert "*.md" in cfg.file_patterns
        reloaded = load_app_config(config_file=config_file)
        assert reloaded.file_patterns == cfg.file_patterns

    def test_config_set_follow_symlinks_true_coerces_bool(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        cfg = set_config_value("follow_symlinks", "true", config_file=config_file)
        assert cfg.follow_symlinks is True

    def test_config_set_follow_symlinks_false_coerces_bool(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        cfg = set_config_value("follow_symlinks", "false", config_file=config_file)
        assert cfg.follow_symlinks is False

    def test_config_set_follow_symlinks_1_coerces_bool(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        cfg = set_config_value("follow_symlinks", "1", config_file=config_file)
        assert cfg.follow_symlinks is True

    def test_config_set_chunk_size_non_numeric_raises_config_error(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        with pytest.raises(ConfigError) as exc_info:
            set_config_value("chunk_size", "abc", config_file=config_file)
        assert "chunk_size" in str(exc_info.value)

    def test_config_set_freshness_bogus_mode_raises_config_error(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        with pytest.raises(ConfigError) as exc_info:
            set_config_value("freshness", "bogus", config_file=config_file)
        assert "freshness" in str(exc_info.value)

    def test_load_app_config_lexical_weight_non_numeric_raises_config_error(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        tomli_w.dump({"schema_version": 1, "lexical_weight": "fast"}, config_file.open("wb"))
        with pytest.raises(ConfigError) as exc_info:
            load_app_config(config_file=config_file)
        assert "lexical_weight" in str(exc_info.value)
