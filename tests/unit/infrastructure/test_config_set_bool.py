from pathlib import Path

import pytest

from simgrep.config import set_config_value
from simgrep.errors import ConfigError


def test_rejects_unrecognized_bool_string(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as exc_info:
        set_config_value("follow_symlinks", "ture", config_file=tmp_path / "config.toml")
    assert "follow_symlinks" in str(exc_info.value)
    assert "expected one of: 1/true/yes/on, 0/false/no/off" in (exc_info.value.hint or "")


@pytest.mark.parametrize("value", ["true", "1", "yes", "on"])
def test_accepts_true_spellings(tmp_path: Path, value: str) -> None:
    cfg = set_config_value("follow_symlinks", value, config_file=tmp_path / "config.toml")
    assert cfg.follow_symlinks is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off"])
def test_accepts_false_spellings(tmp_path: Path, value: str) -> None:
    cfg = set_config_value("follow_symlinks", value, config_file=tmp_path / "config.toml")
    assert cfg.follow_symlinks is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("off", False)],
)
def test_persisted_value_round_trips(tmp_path: Path, raw: str, expected: bool) -> None:
    config_file = tmp_path / "config.toml"
    set_config_value("follow_symlinks", raw, config_file=config_file)
    from simgrep.config import load_app_config

    assert load_app_config(config_file=config_file).follow_symlinks is expected


@pytest.mark.parametrize("non_str", [True, False, 1])
def test_non_string_values_keep_bool_semantics(tmp_path: Path, non_str: bool | int) -> None:
    cfg = set_config_value("follow_symlinks", non_str, config_file=tmp_path / "config.toml")
    assert isinstance(cfg.follow_symlinks, bool)
    assert cfg.follow_symlinks is bool(non_str)
