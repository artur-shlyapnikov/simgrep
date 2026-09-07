from __future__ import annotations

import tomllib
from collections.abc import Callable
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any, cast

import tomli_w

from simgrep.errors import ConfigError
from simgrep.models import SCHEMA_VERSION, AppConfig, ChangeDetectionMode, FreshnessMode, LexicalFallbackMode, ResultFormat

DEFAULT_CONFIG_PATH = Path("~/.config/simgrep/config.toml")


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, FreshnessMode | ChangeDetectionMode | LexicalFallbackMode | ResultFormat):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _serialize(v) for k, v in asdict(cast(Any, value)).items()}
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_serialize(v) for v in value]
    return value


def _coerce_field(key: str, raw: Any, convert: Callable[[Any], Any]) -> Any:
    try:
        return convert(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid config value for '{key}': {raw!r}", hint=str(exc)) from exc


def _validate_config(config: AppConfig) -> None:
    if config.chunk_size < 1:
        raise ConfigError(f"Invalid config value for 'chunk_size': {config.chunk_size!r}", hint="must be >= 1")
    if config.chunk_overlap < 0 or config.chunk_overlap >= config.chunk_size:
        raise ConfigError(
            f"Invalid config value for 'chunk_overlap': {config.chunk_overlap!r}",
            hint="must satisfy 0 <= chunk_overlap < chunk_size",
        )
    if config.batch_size < 1:
        raise ConfigError(f"Invalid config value for 'batch_size': {config.batch_size!r}", hint="must be >= 1")
    if config.lexical_top < 0:
        raise ConfigError(f"Invalid config value for 'lexical_top': {config.lexical_top!r}", hint="must be >= 0")
    if not 0.0 <= config.lexical_weight <= 1.0:
        raise ConfigError(f"Invalid config value for 'lexical_weight': {config.lexical_weight!r}", hint="must be between 0.0 and 1.0")
    if config.context_lines < 0:
        raise ConfigError(f"Invalid config value for 'context_lines': {config.context_lines!r}", hint="must be >= 0")
    if config.max_chars is not None and config.max_chars < 1:
        raise ConfigError(f"Invalid config value for 'max_chars': {config.max_chars!r}", hint="must be >= 1 or null")
    if config.max_file_size_bytes is not None and config.max_file_size_bytes < 1:
        raise ConfigError(f"Invalid config value for 'max_file_size_bytes': {config.max_file_size_bytes!r}", hint="must be >= 1 or null")
    if not config.model.strip():
        raise ConfigError(f"Invalid config value for 'model': {config.model!r}", hint="must be a non-empty model name")
    if not config.file_patterns or not any(p.strip() for p in config.file_patterns):
        raise ConfigError(
            f"Invalid config value for 'file_patterns': {config.file_patterns!r}",
            hint="must be a non-empty pattern list",
        )


def _coerce_config(data: dict[str, Any]) -> AppConfig:
    schema_version = _coerce_field("schema_version", data.get("schema_version", SCHEMA_VERSION), int)
    if schema_version != SCHEMA_VERSION:
        return AppConfig()
    patterns = data.get("file_patterns")
    if patterns is None:
        file_patterns = AppConfig().file_patterns
    elif isinstance(patterns, list | tuple):
        file_patterns = tuple(str(p) for p in patterns)
    else:
        raise ConfigError(f"Invalid config value for 'file_patterns': {patterns!r}", hint="must be a list of glob patterns")
    result = AppConfig(
        schema_version=schema_version,
        model=str(data.get("model", AppConfig().model)),
        chunk_size=_coerce_field("chunk_size", data.get("chunk_size", AppConfig().chunk_size), int),
        chunk_overlap=_coerce_field("chunk_overlap", data.get("chunk_overlap", AppConfig().chunk_overlap), int),
        batch_size=_coerce_field("batch_size", data.get("batch_size", AppConfig().batch_size), int),
        max_file_size_bytes=(
            None
            if data.get("max_file_size_bytes") is None
            else _coerce_field("max_file_size_bytes", data.get("max_file_size_bytes", AppConfig().max_file_size_bytes or 0), int)
        ),
        follow_symlinks=bool(data.get("follow_symlinks", AppConfig().follow_symlinks)),
        file_patterns=file_patterns,
        lexical_top=_coerce_field("lexical_top", data.get("lexical_top", AppConfig().lexical_top), int),
        lexical_weight=_coerce_field("lexical_weight", data.get("lexical_weight", AppConfig().lexical_weight), float),
        freshness=_coerce_field("freshness", str(data.get("freshness", AppConfig().freshness.value)), FreshnessMode),
        context_lines=_coerce_field("context_lines", data.get("context_lines", AppConfig().context_lines), int),
        max_chars=_coerce_field("max_chars", data.get("max_chars", AppConfig().max_chars), int),
    )
    _validate_config(result)
    return result


def load_app_config(config_file: Path | None = None) -> AppConfig:
    path = (config_file or DEFAULT_CONFIG_PATH).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        cfg = AppConfig()
        save_app_config(cfg, config_file=path)
        return cfg
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid config TOML: {path}", hint=str(exc)) from exc
    return _coerce_config(data)


def save_app_config(config: AppConfig, config_file: Path | None = None) -> None:
    path = (config_file or DEFAULT_CONFIG_PATH).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        tomli_w.dump(_serialize(config), handle)


def set_config_value(key: str, value: Any, *, config_file: Path | None = None) -> AppConfig:
    cfg = load_app_config(config_file=config_file)
    if not hasattr(cfg, key):
        raise ConfigError(f"Unknown config key: {key}")
    field_value = getattr(cfg, key)
    try:
        if isinstance(field_value, bool):
            if isinstance(value, str):
                lowered = value.lower()
                if lowered in {"1", "true", "yes", "on"}:
                    coerced: Any = True
                elif lowered in {"0", "false", "no", "off"}:
                    coerced = False
                else:
                    raise ConfigError(
                        f"Invalid config value for '{key}': {value!r}",
                        hint="expected one of: 1/true/yes/on, 0/false/no/off",
                    )
            else:
                coerced = bool(value)
        elif isinstance(field_value, int) and not isinstance(field_value, bool):
            coerced = int(value)
        elif isinstance(field_value, float):
            coerced = float(value)
        elif isinstance(field_value, tuple):
            coerced = tuple(str(part).strip() for part in str(value).split(",") if str(part).strip())
        elif isinstance(field_value, FreshnessMode):
            coerced = FreshnessMode(str(value))
        else:
            coerced = str(value)
    except (TypeError, ValueError) as exc:
        expected = type(field_value).__name__ if not isinstance(field_value, FreshnessMode) else "FreshnessMode (auto/skip/check)"
        raise ConfigError(f"Invalid config value for '{key}': {value!r}", hint=f"expected {expected}") from exc
    updated = replace(cfg, **{key: coerced})
    _validate_config(updated)
    save_app_config(updated, config_file=config_file)
    return updated
