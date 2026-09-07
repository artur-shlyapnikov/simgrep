from __future__ import annotations

from pathlib import Path

import pytest

from simgrep.files import FileScanEntry, build_file_plan, calculate_file_hash
from simgrep.models import ChangeDetectionMode, FileRecord, ScanOptions


def _entry(file_path: Path, rel_path: str) -> FileScanEntry:
    stat = file_path.stat()
    return FileScanEntry(file_path, file_path.resolve(), rel_path, stat.st_size, stat.st_mtime_ns)


def test_new_file_vanished_before_plan_is_controlled_unreadable(tmp_path: Path) -> None:
    missing = tmp_path / "vanished.py"
    discovered = [FileScanEntry(missing, missing.resolve(), "vanished.py", 5, 12345)]

    plan = build_file_plan(discovered, {}, options=ScanOptions(), change_detection=ChangeDetectionMode.hash)

    assert len(plan.entries) == 1
    entry = plan.entries[0]
    assert entry.path == missing.resolve()
    assert entry.status == "unreadable"
    assert entry.reason == "hash_failed"
    assert entry.size_bytes == 5
    assert entry.mtime_ns == 12345


def test_new_file_hash_oserror_is_controlled_unreadable(tmp_path: Path) -> None:
    from unittest.mock import patch

    file_path = tmp_path / "a.py"
    file_path.write_text("alpha", encoding="utf-8")
    discovered = [_entry(file_path, "a.py")]

    with patch("simgrep.files.calculate_file_hash", side_effect=OSError("hash denied")):
        plan = build_file_plan(discovered, {}, options=ScanOptions(), change_detection=ChangeDetectionMode.hash)

    assert len(plan.entries) == 1
    assert plan.entries[0].status == "unreadable"
    assert plan.entries[0].reason == "hash_failed"


def test_existing_new_file_still_gets_new_status_with_hash(tmp_path: Path) -> None:
    file_path = tmp_path / "a.py"
    file_path.write_text("alpha", encoding="utf-8")

    plan = build_file_plan([_entry(file_path, "a.py")], {}, options=ScanOptions(), change_detection=ChangeDetectionMode.hash)

    assert len(plan.entries) == 1
    entry = plan.entries[0]
    assert entry.path == file_path.resolve()
    assert entry.status == "new"
    assert entry.new_hash is not None
    assert len(entry.new_hash) == 64


def test_stat_mode_never_hashes_so_missing_file_cannot_raise(tmp_path: Path) -> None:
    missing = tmp_path / "vanished.py"
    discovered = [FileScanEntry(missing, missing.resolve(), "vanished.py", 5, 12345)]

    # stat mode never hashes, so a vanished new-file entry cannot raise; it
    # falls through to the normal size/mtime comparison against the record.
    plan = build_file_plan(
        discovered,
        {missing.resolve(): FileRecord(id=1, path=missing.resolve(), size_bytes=1, mtime_ns=1, sha256=None)},
        options=ScanOptions(),
        change_detection=ChangeDetectionMode.stat,
    )

    assert len(plan.entries) == 1
    assert plan.entries[0].status == "changed"


@pytest.mark.parametrize("change_detection", [ChangeDetectionMode.hash])
def test_mixed_batch_continues_past_unreadable_new_file(tmp_path: Path, change_detection: ChangeDetectionMode) -> None:
    good = tmp_path / "good.py"
    good.write_text("ok", encoding="utf-8")
    missing = tmp_path / "gone.py"
    discovered = [
        FileScanEntry(missing, missing.resolve(), "gone.py", 5, 999),
        _entry(good, "good.py"),
    ]

    plan = build_file_plan(discovered, {}, options=ScanOptions(), change_detection=change_detection)

    by_name = {entry.path.name: entry for entry in plan.entries}
    assert by_name["gone.py"].status == "unreadable"
    assert by_name["gone.py"].reason == "hash_failed"
    assert by_name["good.py"].status == "new"


def test_existing_file_hash_oserror_yields_unreadable_without_existing_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The EXISTING-file branch predates ae34fdc but was never pinned: hash failure on a
    # known file must be controlled, and today it drops existing_file_id (see design doc Q1).
    file_path = tmp_path / "known.py"
    file_path.write_text("alpha", encoding="utf-8")
    existing = {file_path.resolve(): FileRecord(id=7, path=file_path.resolve(), size_bytes=5, mtime_ns=1, sha256="a" * 64)}

    def deny(_path: Path) -> str:
        raise PermissionError(13, "hash denied")

    monkeypatch.setattr("simgrep.files.calculate_file_hash", deny)
    plan = build_file_plan([_entry(file_path, "known.py")], existing, options=ScanOptions(), change_detection=ChangeDetectionMode.hash)

    assert len(plan.entries) == 1
    entry = plan.entries[0]
    assert entry.status == "unreadable"
    assert entry.reason == "hash_failed"
    assert entry.existing_file_id is None, "Current contract: link to the stored row is lost"
    assert plan.unreadable_count == 1


def test_too_large_precedes_hashing_for_vanished_new_file(tmp_path: Path) -> None:
    # Size gate runs before calculate_file_hash, so an oversized vanished file must be
    # too_large (reason None), never unreadable/hash_failed.
    missing = tmp_path / "big.py"
    discovered = [FileScanEntry(missing, missing.resolve(), "big.py", 10_000, 12345)]

    plan = build_file_plan(
        discovered,
        {},
        options=ScanOptions(max_file_size_bytes=1024),
        change_detection=ChangeDetectionMode.hash,
    )

    assert len(plan.entries) == 1
    assert plan.entries[0].status == "too_large"
    assert plan.entries[0].reason is None


def test_unreadable_new_file_recovers_to_new_on_next_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file_path = tmp_path / "flaky.py"
    file_path.write_text("alpha", encoding="utf-8")
    real_hash = calculate_file_hash
    calls = {"count": 0}

    def flaky(path: Path) -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("transient lock")
        return real_hash(path)

    monkeypatch.setattr("simgrep.files.calculate_file_hash", flaky)

    plan_first = build_file_plan([_entry(file_path, "flaky.py")], {}, options=ScanOptions(), change_detection=ChangeDetectionMode.hash)
    plan_second = build_file_plan([_entry(file_path, "flaky.py")], {}, options=ScanOptions(), change_detection=ChangeDetectionMode.hash)

    assert plan_first.entries[0].status == "unreadable"
    assert plan_second.entries[0].status == "new"
    assert len(plan_second.entries[0].new_hash or "") == 64
