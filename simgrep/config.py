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
        error_message = f"Fatal: Could not create simgrep data directory at '{config.default_project_data_dir}'. " f"Please check permissions. Error: {e}"
        print(error_message, file=sys.stderr)
        raise SimgrepConfigError(error_message) from e

    if config.config_file.exists():
        with open(config.config_file, "rb") as f:
            data = tomllib.load(f)
        config = SimgrepConfig(**data)
        if "default" not in config.projects:
            config.projects["default"] = _create_default_project(config)
            _write_config(config)
    else:
        config.projects["default"] = _create_default_project(config)
        _write_config(config)

    default_proj = config.projects["default"]

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


def _serialize_paths(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _serialize_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_paths(v) for v in obj]
    return obj


def _dumps_toml(data: Dict[str, Any]) -> str:
    base = dict(data)
    projects = base.pop("projects", {})
    lines = []
    for key, value in base.items():
        lines.append(f"{key} = {json.dumps(value)}")
    for name, proj in projects.items():
        lines.append(f"\n[projects.{name}]")
        for k, v in proj.items():
            lines.append(f"{k} = {json.dumps(v)}")
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


def add_project_to_config(config: SimgrepConfig, project_name: str) -> ProjectConfig:
    """Create a new project entry in the config and write the file.

    Parameters
    ----------
    config:
        Existing loaded configuration which will be mutated.
    project_name:
        Name of the project to create.

    Returns
    -------
    ProjectConfig
        The newly created project configuration.
    """

    if project_name in config.projects:
        raise SimgrepConfigError(f"Project '{project_name}' already exists in configuration")

    project_dir = config.db_directory / "projects" / project_name
    project_dir.mkdir(parents=True, exist_ok=True)

    project_cfg = ProjectConfig(
        name=project_name,
        indexed_paths=[],
        embedding_model=config.default_embedding_model_name,
        db_path=project_dir / "metadata.duckdb",
        usearch_index_path=project_dir / "index.usearch",
    )

    config.projects[project_name] = project_cfg
    _write_config(config)
    return project_cfg


def save_config(config: SimgrepConfig) -> None:
    """Persist the given configuration back to ``config.toml``."""
    _write_config(config)