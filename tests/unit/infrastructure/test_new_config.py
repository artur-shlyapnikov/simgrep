from __future__ import annotations

from pathlib import Path

from simgrep.config import load_app_config, set_config_value


def test_load_app_config_autocreates_file(tmp_path: Path) -> None:
    config_file = tmp_path / "cfg" / "config.toml"
    cfg = load_app_config(config_file=config_file)
    assert config_file.exists()
    assert cfg.model == "ibm-granite/granite-embedding-30m-english"
    assert cfg.chunk_size == 128
    assert cfg.batch_size == 128


def test_set_config_value(tmp_path: Path) -> None:
    config_file = tmp_path / "cfg" / "config.toml"
    cfg = set_config_value("chunk_overlap", 64, config_file=config_file)
    assert cfg.chunk_overlap == 64
    reloaded = load_app_config(config_file=config_file)
    assert reloaded.chunk_overlap == 64


def test_set_config_value_coerces_float_and_str_fields(tmp_path: Path) -> None:
    config_file = tmp_path / "cfg.toml"
    cfg = set_config_value("lexical_weight", "0.75", config_file=config_file)
    assert cfg.lexical_weight == 0.75
    cfg = set_config_value("model", "custom-model", config_file=config_file)
    assert cfg.model == "custom-model"
    reloaded = load_app_config(config_file=config_file)
    assert reloaded.lexical_weight == 0.75
    assert reloaded.model == "custom-model"
