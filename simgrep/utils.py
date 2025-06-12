import hashlib
import tomllib
from pathlib import Path
from typing import List, Optional

import pathspec

from .models import SimgrepConfig


def gather_files_to_process(path: Path, patterns: List[str]) -> List[Path]:
    """Return files in ``path`` matching any of the glob ``patterns``.

    Respects ``.gitignore`` entries found in the target directory.
    """
    base_dir = path.parent if path.is_file() else path

    ignore_spec = None
    gitignore_file = base_dir / ".gitignore"
    if gitignore_file.is_file():
        try:
            ignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", gitignore_file.read_text().splitlines())
        except Exception:
            ignore_spec = None

    if path.is_file():
        if ignore_spec and ignore_spec.match_file(path.name):
            return []
        return [path.resolve()]

    found: set[Path] = set()
    for pattern in patterns:
        for p in base_dir.rglob(pattern):
            if not p.is_file():
                continue
            rel = p.relative_to(base_dir)
            if ignore_spec and ignore_spec.match_file(str(rel)):
                continue
            found.add(p.resolve())

    return sorted(found)


def find_project_root(path: Optional[Path] = None) -> Optional[Path]:
    """
    Finds the project root by searching for a '.simgrep' directory
    upwards from the given path.
    """
    current = (path or Path.cwd()).resolve()
    while True:
        if (current / ".simgrep").is_dir():
            return current
        if current.parent == current:  # Reached the filesystem root
            return None
        current = current.parent


def get_project_name_from_local_config(project_root: Path) -> Optional[str]:
    """
    Reads the project name from the .simgrep/config.toml file.
    """
    config_file = project_root / ".simgrep" / "config.toml"
    if not config_file.is_file():
        return None
    try:
        with open(config_file, "rb") as f:
            data = tomllib.load(f)
        return data.get("project_name")
    except (tomllib.TOMLDecodeError, OSError):
        return None


def get_ephemeral_cache_paths(search_root: Path, cfg: SimgrepConfig) -> tuple[Path, Path]:
    """Return paths for ephemeral cache files based on the search root."""
    key = hashlib.sha1(str(search_root.resolve()).encode()).hexdigest()[:12]
    base = cfg.ephemeral_cache_dir / key
    return base / "metadata.duckdb", base / "index.usearch"
