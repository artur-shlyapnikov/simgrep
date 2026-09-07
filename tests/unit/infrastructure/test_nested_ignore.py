"""Regression tests for nested .gitignore / .repo_ignore support in scan_files."""

from __future__ import annotations

from pathlib import Path

from simgrep.files import scan_files
from simgrep.models import ScanOptions


def _rel_names(root: Path) -> set[str]:
    return {entry.rel_path for entry in scan_files(root, ScanOptions(patterns=("*",)))}


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_nested_gitignore_excludes_file_in_subdir(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    _write(root / "keep.py")
    _write(root / "sub" / "noise.tmp")
    _write(root / "sub" / "code.py")
    _write(root / "sub" / ".gitignore", "*.tmp\n")

    names = _rel_names(root)
    assert "sub/noise.tmp" not in names
    assert {"keep.py", "sub/code.py"} <= names


def test_nested_negation_overrides_root_pattern(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    _write(root / ".gitignore", "*.log\n")
    _write(root / "drop.log")
    _write(root / "sub" / "keep.log")
    _write(root / "sub" / "other.log")
    _write(root / "sub" / ".gitignore", "!keep.log\n")

    names = _rel_names(root)
    assert "drop.log" not in names
    assert "sub/other.log" not in names
    assert "sub/keep.log" in names


def test_sibling_subdir_ignore_does_not_leak(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    _write(root / "a" / "secret.tmp")
    _write(root / "a" / ".gitignore", "*.tmp\n")
    _write(root / "b" / "visible.tmp")
    _write(root / "b" / "code.py")

    names = _rel_names(root)
    assert "a/secret.tmp" not in names
    assert {"b/visible.tmp", "b/code.py"} <= names


def test_nested_directory_pruning(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    _write(root / "sub" / ".gitignore", "inner/\n")
    _write(root / "sub" / "inner" / "deep.py")
    _write(root / "sub" / "outer.py")

    names = _rel_names(root)
    assert "sub/inner/deep.py" not in names
    assert {"sub/.gitignore", "sub/outer.py"} <= names


def test_root_only_behavior_unchanged(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    _write(root / ".gitignore", "*.log\n")
    _write(root / "app.log")
    _write(root / "main.py")

    names = _rel_names(root)
    assert "app.log" not in names
    assert "main.py" in names


def test_single_file_input_honors_parent_ignore_spec(tmp_path: Path) -> None:
    # For a single-file path the root is the file's parent directory; its
    # ignore files still apply and no directory walk occurs.
    root = tmp_path / "proj"
    _write(root / ".gitignore", "*.tmp\n")
    _write(root / "data.tmp")

    entries = scan_files(root / "data.tmp", ScanOptions(patterns=("*",)))
    assert entries == []
