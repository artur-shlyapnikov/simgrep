from __future__ import annotations

from pathlib import Path

import pytest

from simgrep.errors import SearchError
from simgrep.indexing import IndexEngine
from simgrep.models import (
    SCHEMA_VERSION,
    AppConfig,
    ChangeDetectionMode,
    FreshnessMode,
    IndexOptions,
    ProjectConfig,
    SearchOptions,
)
from simgrep.search import SearchEngine
from simgrep.store import Store
from tests.conftest import FakeRuntime


def _project(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path,), "fake", 128, 20)


def test_freshness_auto_reindexes_changed_file(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    project = _project(tmp_path)
    (tmp_path / "a.py").write_text("alpha bravo charlie delta", encoding="utf-8")
    app_config = AppConfig(model="fake")

    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True))

    store = Store.open(project.metadata_db_path)
    chunk_before = store._conn.execute("SELECT text FROM chunks").fetchone()
    store.close()
    assert chunk_before is not None
    assert "alpha" in chunk_before[0]

    (tmp_path / "a.py").write_text("foxtrot golf hotel india", encoding="utf-8")

    SearchEngine(fake_runtime).search_project(project, app_config, SearchOptions(query="foxtrot"), FreshnessMode.auto)

    store = Store.open(project.metadata_db_path)
    chunks_after = store._conn.execute("SELECT text FROM chunks").fetchall()
    store.close()
    assert len(chunks_after) == 1
    assert "foxtrot" in chunks_after[0][0]
    assert "alpha" not in chunks_after[0][0]


def test_freshness_auto_deletes_chunks_on_file_removal(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    project = _project(tmp_path)
    (tmp_path / "a.py").write_text("secret token abc123xyz", encoding="utf-8")
    app_config = AppConfig(model="fake")

    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True))

    store_before = Store.open(project.metadata_db_path)
    counts_before = store_before.counts()
    store_before.close()

    (tmp_path / "a.py").unlink()

    outcome = SearchEngine(fake_runtime).search_project(project, app_config, SearchOptions(query="secret"), FreshnessMode.auto)
    assert len(outcome.results) == 0

    store_after = Store.open(project.metadata_db_path)
    counts_after = store_after.counts()
    store_after.close()
    assert counts_after.files_count < counts_before.files_count


def test_freshness_auto_handles_rename_move(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    project = _project(tmp_path)
    (tmp_path / "old_name.py").write_text("find me please now", encoding="utf-8")
    app_config = AppConfig(model="fake")

    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True))

    store = Store.open(project.metadata_db_path)
    old_paths = store.get_files()
    store.close()
    assert any(p.name == "old_name.py" for p in old_paths)

    (tmp_path / "old_name.py").rename(tmp_path / "new_name.py")

    SearchEngine(fake_runtime).search_project(project, app_config, SearchOptions(query="find me"), FreshnessMode.auto)

    store = Store.open(project.metadata_db_path)
    new_paths = store.get_files()
    store.close()
    assert any(p.name == "new_name.py" for p in new_paths)
    assert not any(p.name == "old_name.py" for p in new_paths)


def test_freshness_auto_git_pull_scenario(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    project = _project(tmp_path)
    (tmp_path / "a.py").write_text("original a text content", encoding="utf-8")
    (tmp_path / "b.py").write_text("b file removed content", encoding="utf-8")
    (tmp_path / "c.py").write_text("unchanged c content", encoding="utf-8")
    app_config = AppConfig(model="fake")

    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True))

    (tmp_path / "a.py").write_text("modified a text content", encoding="utf-8")
    (tmp_path / "b.py").unlink()
    (tmp_path / "d.py").write_text("new d file added content", encoding="utf-8")

    SearchEngine(fake_runtime).search_project(project, app_config, SearchOptions(query="modified"), FreshnessMode.auto)

    store = Store.open(project.metadata_db_path)
    files_after = {f.path.name for f in store.get_files().values()}
    chunks_after = store._conn.execute("SELECT label, text FROM chunks ORDER BY label").fetchall()
    store.close()

    assert "a.py" in files_after
    assert "d.py" in files_after
    assert "b.py" not in files_after
    assert "c.py" in files_after

    a_chunk = next((c[1] for c in chunks_after if "modified a" in c[1]), None)
    assert a_chunk is not None
    assert "modified a" in a_chunk


def test_reindex_without_changes_no_new_vectors(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    project = _project(tmp_path)
    (tmp_path / "a.py").write_text("stable content here", encoding="utf-8")
    app_config = AppConfig(model="fake")

    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True))

    store_before = Store.open(project.metadata_db_path)
    files_before = store_before.counts().files_count
    chunks_before = store_before.counts().chunks_count
    store_before.close()

    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions())

    store_after = Store.open(project.metadata_db_path)
    files_after = store_after.counts().files_count
    chunks_after = store_after.counts().chunks_count
    store_after.close()

    assert files_before == files_after
    assert chunks_before == chunks_after

    results_before = SearchEngine(fake_runtime).search_project(project, app_config, SearchOptions(query="stable content"), FreshnessMode.skip)
    results_after = SearchEngine(fake_runtime).search_project(project, app_config, SearchOptions(query="stable content"), FreshnessMode.skip)
    assert len(results_before.results) == len(results_after.results)


def test_freshness_check_raises_on_any_mutation(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    project = _project(tmp_path)
    (tmp_path / "a.py").write_text("content file here", encoding="utf-8")
    app_config = AppConfig(model="fake")

    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True))

    (tmp_path / "a.py").write_text("changed file content", encoding="utf-8")
    with pytest.raises(SearchError, match="stale"):
        SearchEngine(fake_runtime).search_project(project, app_config, SearchOptions(query="changed"), FreshnessMode.check)


def test_freshness_check_raises_on_deleted_only(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    project = _project(tmp_path)
    (tmp_path / "a.py").write_text("to delete file here", encoding="utf-8")
    app_config = AppConfig(model="fake")

    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True))

    (tmp_path / "a.py").unlink()

    with pytest.raises(SearchError, match="stale"):
        SearchEngine(fake_runtime).search_project(project, app_config, SearchOptions(query="to delete"), FreshnessMode.check)


def test_freshness_skip_returns_old_index(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    project = _project(tmp_path)
    (tmp_path / "a.py").write_text("elephant hotel mike", encoding="utf-8")
    app_config = AppConfig(model="fake")

    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True))

    store = Store.open(project.metadata_db_path)
    chunk_before = store._conn.execute("SELECT text FROM chunks").fetchone()
    store.close()
    assert chunk_before is not None
    assert "elephant" in chunk_before[0]

    (tmp_path / "a.py").write_text("juliet kilo lima", encoding="utf-8")

    results = SearchEngine(fake_runtime).search_project(project, app_config, SearchOptions(query="elephant"), FreshnessMode.skip)
    assert len(results.results) == 1
    assert "elephant" in results.results[0].chunk_text

    search_new = SearchEngine(fake_runtime).search_project(project, app_config, SearchOptions(query="juliet"), FreshnessMode.skip)
    assert len(search_new.results) >= 0
    assert search_new.results[0].chunk_text == "elephant hotel mike"


def test_hash_mode_detects_content_change_with_same_size_mtime(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    project = _project(tmp_path)
    (tmp_path / "a.py").write_text("original content here", encoding="utf-8")
    app_config = AppConfig(model="fake")

    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True, change_detection=ChangeDetectionMode.hash))

    (tmp_path / "a.py").write_text("modified content here", encoding="utf-8")

    outcome = SearchEngine(fake_runtime).search_project(
        project,
        app_config,
        SearchOptions(query="modified"),
        FreshnessMode.auto,
    )
    assert len(outcome.results) == 1


def test_stat_mode_unchanged_on_same_size_mtime(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    project = _project(tmp_path)
    (tmp_path / "a.py").write_text("stable content text", encoding="utf-8")
    app_config = AppConfig(model="fake")

    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True, change_detection=ChangeDetectionMode.stat))

    plan1 = IndexEngine(fake_runtime).plan_project(project, app_config, IndexOptions(change_detection=ChangeDetectionMode.stat))
    assert plan1.unchanged_count == 1
    assert plan1.changed_count == 0


def test_indexed_path_change_reflects_in_freshness_plan(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    subdir = tmp_path / "sub"
    subdir.mkdir()
    project = _project(tmp_path)

    (tmp_path / "a.py").write_text("root file content", encoding="utf-8")
    (subdir / "b.py").write_text("sub file content", encoding="utf-8")
    app_config = AppConfig(model="fake")

    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True))

    store = Store.open(project.metadata_db_path)
    files_initial = {f.path.name for f in store.get_files().values()}
    store.close()
    assert "a.py" in files_initial
    assert "b.py" in files_initial

    new_project = ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path / "a.py",), "fake", 128, 20)

    plan = IndexEngine(fake_runtime).plan_project(new_project, app_config, IndexOptions())
    assert plan.deleted_count == 1

    SearchEngine(fake_runtime).search_project(new_project, app_config, SearchOptions(query="sub file content"), FreshnessMode.auto)

    store = Store.open(project.metadata_db_path)
    files_after = {f.path.name for f in store.get_files().values()}
    store.close()
    assert "b.py" not in files_after
    assert "a.py" in files_after


def test_add_path_incremental_indexes_new_path(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    project = _project(tmp_path)
    (tmp_path / "a.py").write_text("original file content", encoding="utf-8")
    app_config = AppConfig(model="fake")

    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True))

    new_dir = tmp_path / "newpath"
    new_dir.mkdir()
    (new_dir / "b.py").write_text("newpath content here", encoding="utf-8")

    updated_project = ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path, new_dir), "fake", 128, 20)

    outcome = SearchEngine(fake_runtime).search_project(updated_project, app_config, SearchOptions(query="newpath content"), FreshnessMode.auto)
    assert len(outcome.results) >= 1


def test_remove_path_shows_deleted_in_plan(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    subdir = tmp_path / "sub"
    subdir.mkdir()
    project = _project(tmp_path)

    (tmp_path / "a.py").write_text("root file content", encoding="utf-8")
    (subdir / "b.py").write_text("sub file content", encoding="utf-8")
    app_config = AppConfig(model="fake")

    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True))

    updated_project = ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path / "a.py",), "fake", 128, 20)

    plan = IndexEngine(fake_runtime).plan_project(updated_project, app_config, IndexOptions())
    deleted_paths = [e.path.name for e in plan.entries if e.status == "deleted"]
    assert "b.py" in deleted_paths
