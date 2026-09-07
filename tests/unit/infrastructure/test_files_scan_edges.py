"""Unit contracts for scan_files/_load_ignore_* error and edge branches.

Every spec pins a defensive-but-reachable branch: unreadable or invalid ignore
specs degrade to no filtering, ignored-directory contents never leak via
symlinks, single-file stat failures yield empty scans, and directory walks
contain per-entry OSErrors without aborting healthy siblings.
"""

from __future__ import annotations

import os
from pathlib import Path

import pathspec
import pytest

from simgrep.files import _load_ignore_spec, _read_ignore_lines, _rewrite_ignore_lines, scan_files, tokenize_text
from simgrep.models import ScanOptions


class _StatFailEntry:
    """os.DirEntry wrapper whose stat() raises OSError."""

    def __init__(self, inner: os.DirEntry[str]) -> None:
        self._inner = inner

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def path(self) -> str:
        return self._inner.path

    def is_symlink(self) -> bool:
        return self._inner.is_symlink()

    def is_dir(self, *, follow_symlinks: bool = False) -> bool:
        return self._inner.is_dir(follow_symlinks=follow_symlinks)

    def is_file(self, *, follow_symlinks: bool = False) -> bool:
        return self._inner.is_file(follow_symlinks=follow_symlinks)

    def stat(self, *, follow_symlinks: bool = False) -> os.stat_result:
        raise OSError(13, "denied")


class _WrappedScandir:
    def __init__(self, entries: list[object]) -> None:
        self._entries = entries

    def __enter__(self) -> object:
        return iter(self._entries)

    def __exit__(self, *exc: object) -> None:
        pass


def _patch_stat_failures(monkeypatch: pytest.MonkeyPatch, failing_names: set[str]) -> None:
    real_scandir = os.scandir

    def flaky_scandir(path: str | os.PathLike[str]) -> _WrappedScandir:
        entries: list[object] = []
        for e in real_scandir(path):
            entries.append(_StatFailEntry(e) if e.name in failing_names else e)
        return _WrappedScandir(entries)

    monkeypatch.setattr(os, "scandir", flaky_scandir)


def _patch_resolve_failures(monkeypatch: pytest.MonkeyPatch, failing_names: set[str]) -> None:
    real_resolve = Path.resolve

    def flaky_resolve(self: Path, *, strict: bool = False) -> Path:
        if self.name in failing_names:
            raise OSError(13, "denied")
        return real_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", flaky_resolve)


class TestFilesScanEdges:
    def test_symlink_into_ignored_directory_is_skipped(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        (root / ".git").mkdir()
        (root / ".git" / "secret.py").write_text("s = 1\n", encoding="utf-8")
        (root / "link.py").symlink_to(root / ".git" / "secret.py")
        (root / "ok.py").write_text("o = 1\n", encoding="utf-8")

        found = {e.path.name for e in scan_files(root, ScanOptions(follow_symlinks=True))}

        assert found == {"ok.py"}

    def test_single_file_stat_oserror_yields_empty_scan(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        target = tmp_path / "app.py"
        target.write_text("x = 1\n", encoding="utf-8")

        def boom(self: Path) -> os.stat_result:
            raise OSError(13, "denied")

        monkeypatch.setattr(Path, "stat", boom)
        monkeypatch.setattr(Path, "is_file", lambda self: True)

        assert scan_files(target, ScanOptions()) == []

    def test_unreadable_gitignore_skipped_while_repo_ignore_still_scopes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        sub = tmp_path / "sub"
        (sub / "junk").mkdir(parents=True)
        (sub / ".gitignore").write_text("junk/\n", encoding="utf-8")
        (sub / ".repo_ignore").write_text("junk/\n", encoding="utf-8")
        (sub / "junk" / "x.py").write_text("x = 1\n", encoding="utf-8")
        (sub / "keep.py").write_text("x = 2\n", encoding="utf-8")

        original_read_text = Path.read_text

        def selective_read_text(self: Path, *args: object, **kwargs: object) -> str:
            if self.name == ".gitignore":
                raise PermissionError(13, "denied")
            return original_read_text(self, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "read_text", selective_read_text)

        assert _read_ignore_lines(sub) == ["junk/"]

        found = {e.path.name for e in scan_files(tmp_path, ScanOptions())}
        assert found == {"keep.py"}

    @pytest.mark.parametrize(
        ("base_rel", "lines", "expected"),
        [
            ("pkg", ["# note", "", "   ", "/build", "logs/", "*.tmp"], ["pkg/build", "pkg/logs/", "pkg/**/*.tmp"]),
            ("", ["# note", "/build"], ["/build"]),
            ("", ["a/b"], ["a/b"]),
            ("pkg", ["!keep"], ["!pkg/**/keep"]),
        ],
    )
    def test_rewrite_ignore_lines_scoping_table(self, base_rel: str, lines: list[str], expected: list[str]) -> None:
        assert _rewrite_ignore_lines(lines, base_rel) == expected
        assert tokenize_text("") == []
        assert tokenize_text("a bb ccc") == ["bb", "ccc"]

    def test_invalid_ignore_spec_degrades_to_no_filtering(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / ".gitignore").write_text("keep.py\n", encoding="utf-8")
        (tmp_path / "keep.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "other.py").write_text("x = 2\n", encoding="utf-8")

        def broken_from_lines(*args: object, **kwargs: object) -> pathspec.PathSpec:
            raise ValueError("bad spec")

        monkeypatch.setattr(pathspec.PathSpec, "from_lines", classmethod(broken_from_lines))

        assert _load_ignore_spec(tmp_path) is None

        found = {e.path.name for e in scan_files(tmp_path, ScanOptions())}
        assert found == {"keep.py", "other.py"}

    def test_walk_tolerates_entry_errors_and_special_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "doomed_dir").mkdir()
        (tmp_path / "doomed_dir" / "nested.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "doomed_file.py").write_text("x = 2\n", encoding="utf-8")
        (tmp_path / "good.py").write_text("x = 3\n", encoding="utf-8")
        os.mkfifo(tmp_path / "pipe")
        _patch_resolve_failures(monkeypatch, {"doomed_dir"})
        _patch_stat_failures(monkeypatch, {"doomed_file.py"})

        found = {e.path.name for e in scan_files(tmp_path, ScanOptions())}

        assert found == {"good.py"}

    def test_walk_aborts_only_unreadable_subtree(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        locked = tmp_path / "locked_dir"
        locked.mkdir()
        (locked / "nested.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "good.py").write_text("x = 2\n", encoding="utf-8")
        real_scandir = os.scandir

        def flaky_scandir(path: str | os.PathLike[str]) -> object:
            if Path(str(path)).name == "locked_dir":
                raise PermissionError(13, "denied")
            return real_scandir(path)

        monkeypatch.setattr(os, "scandir", flaky_scandir)

        found = {e.path.name for e in scan_files(tmp_path, ScanOptions())}

        assert found == {"good.py"}

    def test_symlink_escaping_root_is_tolerated(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        outside = tmp_path / "outside"
        (outside / "deep").mkdir(parents=True)
        (outside / "deep" / "far.py").write_text("f = 1\n", encoding="utf-8")
        root.mkdir()
        (root / "link").symlink_to(outside)
        (root / "near.py").write_text("n = 1\n", encoding="utf-8")

        found = {e.path.name for e in scan_files(root, ScanOptions(follow_symlinks=True))}

        assert found == {"near.py"}
