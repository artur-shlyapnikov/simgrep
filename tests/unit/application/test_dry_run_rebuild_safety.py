from __future__ import annotations

from pathlib import Path

import pytest

from simgrep.indexing import IndexEngine
from simgrep.models import SCHEMA_VERSION, AppConfig, IndexOptions, IndexState, ProjectConfig
from simgrep.store import Store
from tests.conftest import FakeRuntime


def _project(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path,), "fake", 128, 20)


def _seed_project(tmp_path: Path, runtime: FakeRuntime) -> ProjectConfig:
    (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
    (tmp_path / "b.py").write_text("gamma delta", encoding="utf-8")
    project = _project(tmp_path)
    stats = IndexEngine(runtime).index_project(project, AppConfig(model="fake"), IndexOptions())
    assert stats.files_indexed == 2
    return project


class TestDryRunRebuildDoesNotWipeStore:
    def test_rebuild_dry_run_leaves_store_and_state_intact(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        project = _seed_project(tmp_path, fake_runtime)

        stats = IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True, dry_run=True))

        store = Store.open(project.metadata_db_path)
        try:
            assert len(store.get_files()) == 2, f"Dry run must not wipe files, got {store.get_files()}"
            counts = store.counts(project.name)
            assert counts.chunks_count > 0, "Dry run must not delete chunks"
            assert store.get_meta("index_state") == IndexState.ready.value, "Dry run must not change index_state"
        finally:
            store.close()
        # Preview semantics preserved: with rebuild the whole corpus is planned as new.
        assert stats.files_seen == 2
        assert stats.files_skipped_unchanged == 0

    def test_failing_dry_run_does_not_poison_index_state(self, tmp_path: Path, fake_runtime: FakeRuntime, monkeypatch: pytest.MonkeyPatch) -> None:
        from simgrep import indexing as indexing_module

        project = _seed_project(tmp_path, fake_runtime)

        def boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated planning failure")

        monkeypatch.setattr(indexing_module, "build_project_file_plan", boom)
        with pytest.raises(RuntimeError, match="simulated planning failure"):
            IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True, dry_run=True))

        store = Store.open(project.metadata_db_path)
        try:
            assert store.get_meta("index_state") == IndexState.ready.value, "Failing dry run must leave index_state untouched"
            assert store.get_meta("last_index_error") in (None, ""), "Failing dry run must not write last_index_error"
            assert len(store.get_files()) == 2
        finally:
            store.close()

    def test_dry_run_on_never_indexed_project_writes_no_state_meta(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
        project = _project(tmp_path)

        stats = IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(dry_run=True))

        assert stats.files_seen == 1
        store = Store.open(project.metadata_db_path)
        try:
            assert store.get_meta("index_state") is None, "First dry-run must not create index_state"
            assert store.get_meta("last_index_error") is None
            assert store.counts(project.name).files_count == 0
        finally:
            store.close()

    def test_rebuild_dry_run_leaves_vector_artifact_byte_identical(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        project = _seed_project(tmp_path, fake_runtime)
        artifact_before = project.vector_index_path.read_bytes()

        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True, dry_run=True))

        assert project.vector_index_path.exists()
        assert project.vector_index_path.read_bytes() == artifact_before, "Dry-run must not rewrite the vector artifact"

    def test_non_dry_run_rebuild_still_clears_rows_removed_from_disk(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        # Control cell for the `rebuild and not options.dry_run` guard: real rebuild must
        # still wipe rows whose source file vanished (regression guard against over-suppression).
        project = _seed_project(tmp_path, fake_runtime)
        (tmp_path / "b.py").unlink()

        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

        store = Store.open(project.metadata_db_path)
        try:
            assert {p.name for p in store.get_files()} == {"a.py"}
            counts = store.counts(project.name)
            assert counts.files_count == 1
            assert counts.chunks_count == 1
        finally:
            store.close()
