import tomllib
from pathlib import Path
from typing import List, Optional


def gather_files_to_process(path: Path, patterns: List[str]) -> List[Path]:
    """Return files in ``path`` matching any of the glob ``patterns``."""
    if path.is_file():
        return [path]

    found: set[Path] = set()
    for pattern in patterns:
        for p in path.rglob(pattern):
            if p.is_file():
                found.add(p.resolve())

    return sorted(found)


def find_project_root(path: Path = Path.cwd()) -> Optional[Path]:
    """
    Finds the project root by searching for a '.simgrep' directory
    upwards from the given path.
    """
    current = path.resolve()
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