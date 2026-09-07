from __future__ import annotations

import time
from pathlib import Path

from simgrep.indexing import IndexEngine
from simgrep.models import SCHEMA_VERSION, AppConfig, IndexOptions, ProjectConfig
from simgrep.store import Store
from tests.conftest import FakeRuntime


def _project(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path,), "fake", 128, 20)


def test_rebuild_empty_project_creates_vector_index_file(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    project = _project(tmp_path)
    engine = IndexEngine(fake_runtime)

    engine.index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

    assert project.vector_index_path.exists()
    store = Store.open(project.metadata_db_path)
    try:
        assert store.get_meta("index_state") == "ready"
    finally:
        store.close()


def test_first_index_empty_project_creates_vector_index_file(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    project = _project(tmp_path)
    engine = IndexEngine(fake_runtime)

    engine.index_project(project, AppConfig(model="fake"), IndexOptions())

    assert project.vector_index_path.exists()


def test_noop_reindex_does_not_rewrite_existing_vector_file(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
    project = _project(tmp_path)
    engine = IndexEngine(fake_runtime)
    app_config = AppConfig(model="fake")
    engine.index_project(project, app_config, IndexOptions())
    assert project.vector_index_path.exists()
    before_ns = project.vector_index_path.stat().st_mtime_ns
    time.sleep(0.02)

    stats = engine.index_project(project, app_config, IndexOptions())

    assert not stats.index_mutated
    assert project.vector_index_path.exists()
    assert project.vector_index_path.stat().st_mtime_ns == before_ns
    store = Store.open(project.metadata_db_path)
    try:
        assert store.get_meta("index_state") == "ready"
    finally:
        store.close()
