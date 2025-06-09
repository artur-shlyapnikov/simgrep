import json
import sys  # for printing to stderr
import tomllib
from pathlib import Path
from typing import Any, Dict

try:
    from .metadata_db import connect_global_db, get_project_by_name, insert_project
    from .models import ProjectConfig, SimgrepConfig
except ImportError:
    from simgrep.metadata_db import connect_global_db, get_project_by_name, insert_project  # type: ignore
    from simgrep.models import ProjectConfig, SimgrepConfig  # type: ignore


# default number of top results to fetch for searches
DEFAULT_K_RESULTS = 5


class SimgrepConfigError(Exception):
    """Custom exception for simgrep configuration errors."""

    pass


def load_or_create_global_config() -> SimgrepConfig:
    """
    Loads simgrep config from TOML file, or creates a default one.
    Ensures data directories and the global database with a 'default' project exist.
    The SimgrepConfig object returned contains global defaults, not project-specific data.

    Returns:
        An instance of SimgrepConfig.

    Raises:
        SimgrepConfigError: If the data directory cannot be created.
    """
    config = SimgrepConfig()

    try:
        config.db_directory.mkdir(parents=True, exist_ok=True)
        config.default_project_data_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        error_message = f"Fatal: Could not create simgrep data directory at '{config.db_directory}'. " f"Please check permissions. Error: {e}"
        print(error_message, file=sys.stderr)
        raise SimgrepConfigError(error_message) from e

    if config.config_file.exists():
        with open(config.config_file, "rb") as f:
            data = tomllib.load(f)
        config = SimgrepConfig(**data)
    else:
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

    return config


def _serialize_paths(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _serialize_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_paths(v) for v in obj]
    return obj


def _dumps_toml(data: Dict[str, Any]) -> str:
    lines = []
    for key, value in data.items():
        lines.append(f"{key} = {json.dumps(value)}")
    return "\n".join(lines) + "\n"


def _write_config(config: SimgrepConfig) -> None:
    data = _serialize_paths(config.model_dump())
    config.config_file.write_text(_dumps_toml(data), encoding="utf-8")


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
