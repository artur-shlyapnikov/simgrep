from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import tomli_w

from simgrep.errors import ProjectError
from simgrep.models import SCHEMA_VERSION, AppConfig, ProjectConfig


def project_file(root: Path) -> Path:
    return root / ".simgrep" / "project.toml"


def _rel_or_abs(root: Path, path: Path, *, allow_outside_root: bool) -> str:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix() or "."
    except ValueError:
        if not allow_outside_root:
            raise ProjectError(f"Path outside project root: {path}")
        return str(resolved_path)


def _resolve_indexed_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def init_project(root: Path, app_config: AppConfig | None = None, *, name: str | None = None, yes: bool = False) -> ProjectConfig:
    root = root.resolve()
    if not root.exists() or not root.is_dir():
        raise ProjectError(f"Project root must be an existing directory: {root}")
    file_path = project_file(root)
    if file_path.exists() and not yes:
        raise ProjectError(f"Project already initialized: {root}", hint="Use --yes to overwrite project.toml.")
    app = app_config or AppConfig()
    cfg = ProjectConfig(
        schema_version=SCHEMA_VERSION,
        name=name or root.name,
        root=root,
        indexed_paths=(root,),
        model=app.model,
        chunk_size=app.chunk_size,
        chunk_overlap=app.chunk_overlap,
    )
    save_project_config(cfg)
    return cfg


def _coerce_int(key: str, raw: Any) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ProjectError(f"Invalid project.toml value for '{key}': {raw!r}", hint=str(exc)) from exc


def load_project_config(root: Path) -> ProjectConfig:
    root = root.resolve()
    file_path = project_file(root)
    if not file_path.exists():
        raise ProjectError(f"Project config not found: {file_path}")
    try:
        with file_path.open("rb") as handle:
            data: dict[str, Any] = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ProjectError(f"Invalid project TOML: {file_path}", hint=str(exc)) from exc
    schema_version = _coerce_int("schema_version", data.get("schema_version", SCHEMA_VERSION))
    if schema_version != SCHEMA_VERSION:
        raise ProjectError(f"Unsupported project schema {schema_version}; expected {SCHEMA_VERSION}.")
    raw_paths = data.get("indexed_paths", ["."])
    if not isinstance(raw_paths, list):
        raise ProjectError("project.toml indexed_paths must be a list.")
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw in raw_paths:
        path = _resolve_indexed_path(root, str(raw))
        if path not in seen:
            seen.add(path)
            resolved.append(path)
    if not resolved:
        raise ProjectError(
            "project.toml indexed_paths must not be empty",
            hint="fix or delete .simgrep/project.toml, or run `simgrep init --yes` to re-create it",
        )
    model = str(data.get("model", AppConfig().model))
    chunk_size = _coerce_int("chunk_size", data.get("chunk_size", AppConfig().chunk_size))
    chunk_overlap = _coerce_int("chunk_overlap", data.get("chunk_overlap", AppConfig().chunk_overlap))
    if not model.strip():
        raise ProjectError(f"Invalid project config value for 'model': {model!r}", hint="must be a non-empty model name")
    if chunk_size < 1:
        raise ProjectError(f"Invalid project config value for 'chunk_size': {chunk_size!r}", hint="must be >= 1")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ProjectError(
            f"Invalid project config value for 'chunk_overlap': {chunk_overlap!r}",
            hint="must satisfy 0 <= chunk_overlap < chunk_size",
        )
    return ProjectConfig(
        schema_version=schema_version,
        name=str(data.get("name", root.name)),
        root=root,
        indexed_paths=tuple(resolved),
        model=model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def save_project_config(config: ProjectConfig) -> None:
    config.simgrep_dir.mkdir(parents=True, exist_ok=True)
    indexed_paths = [_rel_or_abs(config.root, p, allow_outside_root=True) for p in config.indexed_paths]
    data = {
        "schema_version": config.schema_version,
        "name": config.name,
        "model": config.model,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "indexed_paths": indexed_paths,
    }
    with project_file(config.root).open("wb") as handle:
        tomli_w.dump(data, handle)


def find_active_project(start: Path | None = None) -> ProjectConfig | None:
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if project_file(candidate).exists():
            return load_project_config(candidate)
    return None


def require_active_project(start: Path | None = None) -> ProjectConfig:
    project = find_active_project(start)
    if project is None:
        raise ProjectError("No simgrep project found.", hint="Run `simgrep init` in the project root.")
    return project


def add_indexed_path(config: ProjectConfig, path: Path, *, allow_outside_root: bool = False) -> ProjectConfig:
    normalized = _rel_or_abs(config.root, path, allow_outside_root=allow_outside_root)
    resolved = _resolve_indexed_path(config.root, normalized)
    paths = list(config.indexed_paths)
    if resolved not in {p.resolve() for p in paths}:
        paths.append(resolved)
    updated = ProjectConfig(
        schema_version=config.schema_version,
        name=config.name,
        root=config.root,
        indexed_paths=tuple(paths),
        model=config.model,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    save_project_config(updated)
    return updated


def remove_indexed_path(config: ProjectConfig, path: Path, *, allow_outside_root: bool = False) -> ProjectConfig:
    normalized = _rel_or_abs(config.root, path, allow_outside_root=allow_outside_root)
    resolved = _resolve_indexed_path(config.root, normalized)
    paths = tuple(p for p in config.indexed_paths if p.resolve() != resolved)
    if not paths:
        raise ProjectError(f"Cannot remove the last indexed path: {resolved}", hint="add another path first, or run `simgrep reset` to remove the project")
    updated = ProjectConfig(
        schema_version=config.schema_version,
        name=config.name,
        root=config.root,
        indexed_paths=paths,
        model=config.model,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    save_project_config(updated)
    return updated


def project_covers_path(config: ProjectConfig, path: Path) -> bool:
    resolved = path.resolve()
    for indexed in config.indexed_paths:
        root = indexed.resolve()
        if root.is_file() and resolved == root:
            return True
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False
