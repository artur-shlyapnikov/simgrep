import hashlib
import tomllib
from pathlib import Path
from typing import List, Optional

import pathspec


def gather_files_to_process(path: Path, patterns: List[str]) -> List[Path]:
    """Return files in ``path`` matching any of the glob ``patterns``.

    Respects ``.gitignore`` entries found in the target directory.
    """
    import fnmatch
    import os

    base_dir = path.parent if path.is_file() else path

    ignore_spec = None
    gitignore_file = base_dir / ".gitignore"
    if gitignore_file.is_file():
        try:
            ignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", gitignore_file.read_text().splitlines())
        except Exception:
            ignore_spec = None

    if path.is_file():
        # If a direct file path is provided, we process it regardless of gitignore.
        # The user's explicit intent overrides the ignore file.
        return [path.resolve()]

    found: set[Path] = set()
    for root, _, files in os.walk(base_dir, followlinks=True):
        root_path = Path(root)
        for name in files:
            file_path = root_path / name
            if not file_path.is_file():
                continue

            rel = file_path.relative_to(base_dir)
            if ignore_spec and ignore_spec.match_file(str(rel)):
                continue

            if any(fnmatch.fnmatch(name, p) for p in patterns):
                found.add(file_path.resolve())

    return sorted(list(found))


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


def calculate_file_hash(file_path: Path) -> str:
    """Compute the SHA256 hash of a file's contents."""
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"File not found or is not a file: {file_path}")

    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except OSError as e:
        raise IOError(f"Error reading file for hashing: {e}") from e


def get_ephemeral_cache_paths(
    target_path: Path, cfg: "SimgrepConfig", patterns: Optional[List[str]] = None
) -> tuple[Path, Path]:
    """Return the DB and index paths for an ephemeral search target."""
    from hashlib import sha256

    resolved = str(target_path.resolve())
    pattern_key = "|".join(sorted(patterns or []))
    digest = sha256(f"{resolved}:{pattern_key}".encode()).hexdigest()[:16]
    base = cfg.ephemeral_cache_dir / digest
    db_path = base / "metadata.duckdb"
    index_path = base / "index.usearch"
    return db_path, index_path
