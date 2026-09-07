from pathlib import Path

import pytest

from simgrep.config import _coerce_config, load_app_config
from simgrep.errors import ConfigError
from simgrep.models import AppConfig


def test_string_file_patterns_raises() -> None:
    with pytest.raises(ConfigError) as excinfo:
        _coerce_config({"file_patterns": "*.py"})
    assert excinfo.value.hint is not None and "list" in excinfo.value.hint


def test_non_list_file_patterns_raises() -> None:
    with pytest.raises(ConfigError):
        _coerce_config({"file_patterns": 42})
    with pytest.raises(ConfigError):
        _coerce_config({"file_patterns": {"a": 1}})


def test_absent_file_patterns_keeps_defaults() -> None:
    config = _coerce_config({})
    assert config.file_patterns == AppConfig().file_patterns


def test_valid_list_file_patterns_preserved() -> None:
    config = _coerce_config({"file_patterns": ["*.py", "*.md"]})
    assert config.file_patterns == ("*.py", "*.md")


def test_load_app_config_rejects_string_patterns(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text('file_patterns = "*.py"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_app_config(config_file=config_file)
