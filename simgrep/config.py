import sys  # for printing to stderr
from pathlib import Path
import tomllib

import tomli_w

try:
    from .models import ProjectConfig, SimgrepConfig
except ImportError:
    from simgrep.models import ProjectConfig, SimgrepConfig


# default number of top results to fetch for searches
DEFAULT_K_RESULTS = 5


class SimgrepConfigError(Exception):
    """Custom exception for simgrep configuration errors."""

    pass


def load_or_create_global_config() -> SimgrepConfig:
    """
    Instantiates a SimgrepConfig object with default values and ensures
    the default project's data directory exists.

    for deliverable 3.1, no toml file reading/writing is performed.

    Returns:
        An instance of SimgrepConfig.

    Raises:
        SimgrepConfigError: If the data directory cannot be created.
    """
    config = SimgrepConfig()

    # ensure directories exist
    try:
        config.db_directory.mkdir(parents=True, exist_ok=True)
        config.default_project_data_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        error_message = (
            f"Fatal: Could not create simgrep data directory at '{config.default_project_data_dir}'. "
            f"Please check permissions. Error: {e}"
        )
        print(error_message, file=sys.stderr)
        raise SimgrepConfigError(error_message) from e

    if config.config_file.exists():
        with open(config.config_file, "rb") as f:
            data = tomllib.load(f)
        config = SimgrepConfig(**data)
        default_proj = config.projects.get("default")
        if default_proj is None:
            default_proj = _create_default_project(config)
            config.projects["default"] = default_proj
            _write_config(config)
    else:
        default_proj = _create_default_project(config)
        config.projects = {"default": default_proj}
        _write_config(config)

    from .metadata_db import connect_global_db, get_project_by_name, insert_project

    global_db_path = config.db_directory / "global_metadata.duckdb"
    conn = connect_global_db(global_db_path)
    try:
        if get_project_by_name(conn, "default") is None:
            insert_project(
                conn,
                project_name="default",
                db_path=str(default_proj.db_path),
                usearch_index_path=str(default_proj.usearch_index_path),
                embedding_model_name=default_proj.embedding_model,
            )
    finally:
        conn.close()

    return config


def _serialize_paths(obj):
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


def _create_default_project(config: SimgrepConfig):
    return ProjectConfig(
        name="default",
        indexed_paths=[],
        embedding_model=config.default_embedding_model_name,
        db_path=config.default_project_data_dir / "metadata.duckdb",
        usearch_index_path=config.default_project_data_dir / "index.usearch",
    )
