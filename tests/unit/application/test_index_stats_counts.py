from __future__ import annotations

from pathlib import Path

from simgrep.indexing import IndexEngine
from simgrep.models import SCHEMA_VERSION, AppConfig, IndexOptions, IndexStats, ProjectConfig
from tests.conftest import FakeRuntime


def _project(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path,), "fake", 128, 20)


class TestDeletedFilesCount:
    def test_files_pruned_deleted_counts_each_deleted_file_once(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
        (tmp_path / "b.py").write_text("gamma delta", encoding="utf-8")
        project = _project(tmp_path)
        engine = IndexEngine(fake_runtime)
        engine.index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

        (tmp_path / "a.py").unlink()
        stats: IndexStats = engine.index_project(project, AppConfig(model="fake"), IndexOptions())

        assert stats.files_pruned_deleted == 1
