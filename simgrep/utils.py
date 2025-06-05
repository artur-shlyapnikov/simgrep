from pathlib import Path
from typing import List


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
