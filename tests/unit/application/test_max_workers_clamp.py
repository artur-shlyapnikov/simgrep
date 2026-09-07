from __future__ import annotations

from pathlib import Path

from simgrep.indexing import IndexEngine
from simgrep.models import SCHEMA_VERSION, AppConfig, IndexOptions, IndexStats, ProjectConfig
from tests.conftest import FakeRuntime


def _project(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path,), "fake", 128, 20)


class TestMaxWorkersClamp:
    def test_max_workers_zero_completes_indexing(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
        (tmp_path / "b.py").write_text("gamma delta", encoding="utf-8")
        project = _project(tmp_path)
        engine = IndexEngine(fake_runtime)

        stats: IndexStats = engine.index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True, max_workers=0))

        assert stats.files_indexed == 2
        assert stats.errors == 0

    # Pins current silent float truncation (int()) — product question PQ2 open.
    def test_float_max_workers_below_one_clamped_to_one(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
        project = _project(tmp_path)
        engine = IndexEngine(fake_runtime)

        stats: IndexStats = engine.index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True, max_workers=0.5))  # type: ignore[arg-type]

        assert stats.files_indexed == 1
        assert stats.errors == 0

    def test_very_large_max_workers_still_indexes(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
        project = _project(tmp_path)
        engine = IndexEngine(fake_runtime)

        stats: IndexStats = engine.index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True, max_workers=10**6))

        assert stats.files_indexed == 1
        assert stats.errors == 0
