"""Range validation of AppConfig values during config loading and mutation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from simgrep.config import load_app_config, save_app_config, set_config_value
from simgrep.errors import ConfigError
from simgrep.models import AppConfig


def _write_config(path: Path, **overrides: object) -> None:
    lines = []
    for key, value in overrides.items():
        if value is None:
            continue
        lines.append(f"{key} = {value!r}")
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("chunk_size", 0),
        ("chunk_size", -5),
        ("chunk_overlap", -1),
        ("chunk_overlap", 128),  # equal to chunk_size
        ("batch_size", 0),
        ("lexical_top", -1),
        ("lexical_weight", -0.1),
        ("lexical_weight", 1.5),
        ("context_lines", -2),
        ("max_chars", 0),
        ("max_file_size_bytes", 0),
    ],
)
def test_invalid_value_raises_from_load(tmp_path: Path, key: str, value: float) -> None:
    cfg_file = tmp_path / "config.toml"
    _write_config(cfg_file, **{key: value})
    with pytest.raises(ConfigError, match=f"'{key}'"):
        load_app_config(config_file=cfg_file)


def test_chunk_overlap_equal_to_chunk_size_raises(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    _write_config(cfg_file, chunk_size=64, chunk_overlap=64)
    with pytest.raises(ConfigError, match="chunk_overlap"):
        load_app_config(config_file=cfg_file)


def test_set_config_value_rejects_and_leaves_file_untouched(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    save_app_config(AppConfig(), config_file=cfg_file)
    before = cfg_file.read_text(encoding="utf-8")
    with pytest.raises(ConfigError, match="chunk_overlap"):
        set_config_value("chunk_overlap", 999, config_file=cfg_file)
    assert cfg_file.read_text(encoding="utf-8") == before


def test_defaults_load_fine(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    assert load_app_config(config_file=cfg_file) == AppConfig()


def test_valid_custom_config_loads(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    _write_config(
        cfg_file,
        chunk_size=512,
        chunk_overlap=64,
        batch_size=16,
        lexical_top=10,
        lexical_weight=0.75,
        context_lines=3,
        max_chars=2000,
        max_file_size_bytes=1024,
    )
    expected = replace(
        AppConfig(),
        chunk_size=512,
        chunk_overlap=64,
        batch_size=16,
        lexical_top=10,
        lexical_weight=0.75,
        context_lines=3,
        max_chars=2000,
        max_file_size_bytes=1024,
    )
    assert load_app_config(config_file=cfg_file) == expected


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_model_rejected_by_set_config_value(tmp_path: Path, value: str) -> None:
    cfg_file = tmp_path / "config.toml"
    save_app_config(AppConfig(), config_file=cfg_file)
    with pytest.raises(ConfigError, match="model"):
        set_config_value("model", value, config_file=cfg_file)


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_model_rejected_from_load(tmp_path: Path, value: str) -> None:
    cfg_file = tmp_path / "config.toml"
    _write_config(cfg_file, **{"model": value})
    with pytest.raises(ConfigError, match="'model'"):
        load_app_config(config_file=cfg_file)


def test_valid_model_passes_through() -> None:
    assert set_config_value("model", "ibm-granite/granite-embedding-30m-english").model == ("ibm-granite/granite-embedding-30m-english")


def test_empty_file_patterns_rejected_by_set_config_value(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    save_app_config(AppConfig(), config_file=cfg_file)
    before = cfg_file.read_text(encoding="utf-8")
    with pytest.raises(ConfigError, match="'file_patterns'"):
        set_config_value("file_patterns", "", config_file=cfg_file)
    assert cfg_file.read_text(encoding="utf-8") == before


def test_blank_file_patterns_rejected_by_set_config_value(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    save_app_config(AppConfig(), config_file=cfg_file)
    with pytest.raises(ConfigError, match="'file_patterns'"):
        set_config_value("file_patterns", " , ", config_file=cfg_file)


def test_empty_file_patterns_list_rejected_from_load(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    _write_config(cfg_file, file_patterns=[])
    with pytest.raises(ConfigError, match="'file_patterns'"):
        load_app_config(config_file=cfg_file)


def test_trailing_comma_file_patterns_accepted_by_set_config_value(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    save_app_config(AppConfig(), config_file=cfg_file)
    updated = set_config_value("file_patterns", "*.py,", config_file=cfg_file)
    assert updated.file_patterns == ("*.py",)


def test_valid_file_patterns_pass_through(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    save_app_config(AppConfig(), config_file=cfg_file)
    updated = set_config_value("file_patterns", "*.py, *.md", config_file=cfg_file)
    assert updated.file_patterns == ("*.py", "*.md")
