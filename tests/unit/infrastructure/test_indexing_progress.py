from __future__ import annotations

from pathlib import Path

import pytest

from simgrep.indexing import IndexEngine, progress_scope
from simgrep.models import SCHEMA_VERSION, AppConfig, IndexOptions, ProjectConfig
from tests.conftest import FakeRuntime


class RecordingReporter:
    def __init__(self) -> None:
        self.phases: list[str] = []
        self.ticks: list[str] = []

    def phase(self, message: str) -> None:
        self.phases.append(message)

    def tick(self, message: str) -> None:
        self.ticks.append(message)


def _project(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path,), "fake", 128, 20)


def test_index_project_reports_scan_and_embed_phases(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    (tmp_path / "a.py").write_text("rollback payment", encoding="utf-8")
    (tmp_path / "b.py").write_text("refund ledger", encoding="utf-8")
    reporter = RecordingReporter()
    with progress_scope(reporter):
        stats = IndexEngine(fake_runtime).index_project(_project(tmp_path), AppConfig(model="fake"), IndexOptions())
    assert stats.files_indexed == 2
    assert any("scan" in phase.lower() for phase in reporter.phases)
    assert any("index" in phase.lower() or "embed" in phase.lower() for phase in reporter.phases)
    assert any("saving" in phase.lower() for phase in reporter.phases)
    # Per-file ticks must reference processed counts.
    assert any("/2" in tick for tick in reporter.ticks)


def test_index_project_without_reporter_stays_silent(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    (tmp_path / "a.py").write_text("rollback payment", encoding="utf-8")
    reporter = RecordingReporter()
    with progress_scope(reporter):
        pass
    with progress_scope(None):
        IndexEngine(fake_runtime).index_project(_project(tmp_path), AppConfig(model="fake"), IndexOptions())
    assert reporter.phases == []
    assert reporter.ticks == []


def test_build_ephemeral_reports_file_counts(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    (tmp_path / "a.py").write_text("charge capture flow", encoding="utf-8")
    reporter = RecordingReporter()
    engine = IndexEngine(fake_runtime)
    with progress_scope(reporter):
        reader = engine.build_ephemeral([tmp_path], AppConfig(model="fake"))
    try:
        assert reader.chunk_count > 0
        assert any("1 file" in phase or "1/" in phase for phase in (*reporter.phases, *reporter.ticks))
    finally:
        reader.close()


def test_reporter_exceptions_never_break_indexing(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    (tmp_path / "a.py").write_text("rollback payment", encoding="utf-8")

    class ExplodingReporter:
        def phase(self, message: str) -> None:
            raise RuntimeError("boom")

        def tick(self, message: str) -> None:
            raise RuntimeError("boom")

    with progress_scope(ExplodingReporter()):
        stats = IndexEngine(fake_runtime).index_project(_project(tmp_path), AppConfig(model="fake"), IndexOptions())
    assert stats.files_indexed == 1


def test_nested_progress_scope_restores_previous(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    outer = RecordingReporter()
    inner = RecordingReporter()
    (tmp_path / "a.py").write_text("rollback payment", encoding="utf-8")
    with progress_scope(outer):
        with progress_scope(inner):
            IndexEngine(fake_runtime).index_project(_project(tmp_path), AppConfig(model="fake"), IndexOptions())
        assert inner.phases, "inner reporter must receive events"
        assert outer.phases == []
        from simgrep.indexing import _current_reporter

        assert _current_reporter() is outer


@pytest.mark.parametrize("missing", ["phase", "tick"])
def test_reporter_missing_method_is_tolerated(tmp_path: Path, fake_runtime: FakeRuntime, missing: str) -> None:
    (tmp_path / "a.py").write_text("rollback payment", encoding="utf-8")

    class HalfReporter:
        pass

    if missing != "phase":
        HalfReporter.phase = lambda self, message: None  # type: ignore[attr-defined,method-assign]
    if missing != "tick":
        HalfReporter.tick = lambda self, message: None  # type: ignore[attr-defined,method-assign]

    with progress_scope(HalfReporter()):
        stats = IndexEngine(fake_runtime).index_project(_project(tmp_path), AppConfig(model="fake"), IndexOptions())
    assert stats.files_indexed == 1
