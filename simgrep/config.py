import sys
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

try:
    from .core.models import ProjectConfig, SimgrepConfig
    from .metadata_db import connect_global_db, get_project_by_name, insert_project
except ImportError:
    from simgrep.core.models import ProjectConfig, SimgrepConfig  # type: ignore
    from simgrep.metadata_db import connect_global_db, get_project_by_name, insert_project  # type: ignore


DEFAULT_K_RESULTS = 5


class SimgrepConfigError(Exception):
    """Custom exception for simgrep configuration errors."""

    pass


def load_global_config() -> SimgrepConfig:
    """
    Loads simgrep config from TOML file.

    Returns:
        An instance of SimgrepConfig.

    Raises:
        SimgrepConfigError: If the config file does not exist.
    """
    config = SimgrepConfig()
    if not config.config_file.exists():
        raise SimgrepConfigError("Global config not found. Please run 'simgrep init --global' to create it.")

    with open(config.config_file, "rb") as f:
        data = tomllib.load(f)
    return SimgrepConfig(**data)


def initialize_global_config(overwrite: bool = False) -> None:
    """
    Creates the global simgrep configuration files and directories.
    """
    config = SimgrepConfig()
    if config.config_file.exists() and not overwrite:
        # This case will be handled in the CLI, but good to have a safeguard.
        return

    try:
        config.db_directory.mkdir(parents=True, exist_ok=True)
        config.default_project_data_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        error_message = f"Fatal: Could not create simgrep data directory at '{config.db_directory}'. Please check permissions. Error: {e}"
        print(error_message, file=sys.stderr)
        raise SimgrepConfigError(error_message) from e

    _write_config(config)

    global_db_path = config.db_directory / "global_metadata.duckdb"
    conn = connect_global_db(global_db_path)
    try:
        if get_project_by_name(conn, "default") is None:
            default_proj_config = _create_default_project(config)
            insert_project(
                conn,
                project_name="default",
                db_path=str(default_proj_config.db_path),
                usearch_index_path=str(default_proj_config.usearch_index_path),
                embedding_model_name=default_proj_config.embedding_model,
            )
    finally:
        conn.close()


def _serialize_paths(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _serialize_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_paths(v) for v in obj]
    return obj


def _write_config(config: SimgrepConfig) -> None:
    data = _serialize_paths(config.model_dump())
    with open(config.config_file, "wb") as f:
        tomli_w.dump(data, f)


def _create_default_project(config: SimgrepConfig) -> ProjectConfig:
    return ProjectConfig(
        name="default",
        indexed_paths=[],
        embedding_model=config.default_embedding_model_name,
        db_path=config.default_project_data_dir / "metadata.duckdb",
        usearch_index_path=config.default_project_data_dir / "index.usearch",
    )


def save_config(config: SimgrepConfig) -> None:
    """Persist the given configuration back to ``config.toml``."""
    _write_config(config)