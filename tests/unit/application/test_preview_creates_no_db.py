from __future__ import annotations

from pathlib import Path

from simgrep.indexing import IndexEngine
from simgrep.models import SCHEMA_VERSION, AppConfig, IndexOptions, ProjectConfig
from tests.conftest import FakeRuntime


def _project(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path,), "fake", 128, 20)


class TestPreviewCreatesNoDb:
    def test_dry_run_on_fresh_project_creates_no_db(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
        (tmp_path / "b.py").write_text("bravo charlie", encoding="utf-8")
        project = _project(tmp_path)
        stats = IndexEngine(FakeRuntime()).index_project(project, AppConfig(model="fake"), IndexOptions(dry_run=True))
        assert not project.metadata_db_path.exists()
        assert not project.vector_index_path.exists()
        assert stats.files_indexed == 2

    def test_plan_project_on_never_indexed_project_creates_no_db(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
        project = _project(tmp_path)
        plan = IndexEngine(FakeRuntime()).plan_project(project, AppConfig(model="fake"))
        assert not project.metadata_db_path.exists()
        new_entries = [e for e in plan.entries if e.status == "new"]
        assert len(new_entries) == 1
        assert plan.new_count == 1

    def test_plan_project_rebuild_skips_read_but_keeps_db(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
        project = _project(tmp_path)
        engine = IndexEngine(FakeRuntime())
        engine.index_project(project, AppConfig(model="fake"), IndexOptions())
        assert project.metadata_db_path.exists()

        plan = engine.plan_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))
        assert project.metadata_db_path.exists()
        assert plan.new_count == 1
