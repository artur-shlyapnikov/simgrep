from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from simgrep.errors import SimgrepError
from simgrep.files import scan_files
from simgrep.models import AppConfig, ScanOptions
from tests.conftest import FakeRuntime


def _options() -> ScanOptions:
    return ScanOptions()


def test_unreadable_root_raises(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("running as root ignores permissions")
    root = tmp_path / "locked"
    (root / "sub").mkdir(parents=True)
    (root / "readable.txt").write_text("x")
    root.chmod(0o000)
    try:
        with pytest.raises(SimgrepError) as exc_info:
            scan_files(root, _options())
        assert "Cannot read directory" in str(exc_info.value)
    finally:
        root.chmod(stat.S_IRWXU)


def test_unreadable_subdir_is_swallowed(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("running as root ignores permissions")
    root = tmp_path
    (root / "keep.txt").write_text("keep")
    locked = root / "locked"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        entries = scan_files(root, _options())
        assert [entry.path.name for entry in entries] == ["keep.txt"]
    finally:
        locked.chmod(stat.S_IRWXU)


def test_single_file_mode_unchanged(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("content")
    entries = scan_files(target, _options())
    assert len(entries) == 1
    assert entries[0].resolved_path == target.resolve()


def test_diff_engine_unreadable_dir_propagates(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("running as root ignores permissions")
    from simgrep.diff_engine import DiffEngine

    readable = tmp_path / "other"
    readable.mkdir()
    (readable / "a.txt").write_text("a\n")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        engine = DiffEngine(FakeRuntime())
        with pytest.raises(SimgrepError):
            engine.diff_paths(locked, readable, AppConfig(model="fake"))
    finally:
        locked.chmod(stat.S_IRWXU)


def test_readable_root_still_scans(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "a.txt").write_text("a")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("c")
    entries = scan_files(tmp_path, _options())
    assert sorted(entry.path.name for entry in entries) == ["a.txt", "b.txt", "c.txt"]


def test_missing_root_reports_path_not_found(tmp_path: Path) -> None:
    with pytest.raises(SimgrepError) as exc_info:
        scan_files(tmp_path / "does-not-exist", _options())
    assert "Path not found" in str(exc_info.value)
    assert exc_info.value.hint is not None
    assert "path" in exc_info.value.hint.lower()


def test_broken_symlink_root_reports_path_not_found(tmp_path: Path) -> None:
    root = tmp_path / "broken-link"
    root.symlink_to(tmp_path / "missing-target")
    with pytest.raises(SimgrepError) as exc_info:
        scan_files(root, _options())
    assert "Path not found" in str(exc_info.value)
    assert exc_info.value.hint is not None
    assert "path" in exc_info.value.hint.lower()
