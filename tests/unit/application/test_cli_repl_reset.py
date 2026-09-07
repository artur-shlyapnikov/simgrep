"""Unit tests for the CLI repl loop's error resilience and reset's DuckDB
sidecar cleanup. Drives the command functions directly with monkeypatched
collaborators; no typer CliRunner."""

from __future__ import annotations

from pathlib import Path

import pytest

import simgrep.main as main_module
from simgrep.errors import SearchError
from simgrep.models import SCHEMA_VERSION, AppConfig, ProjectConfig
from simgrep.store import Store


def _project(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(
        schema_version=SCHEMA_VERSION,
        name="test-project",
        root=tmp_path,
        indexed_paths=(tmp_path,),
        model=AppConfig().model,
        chunk_size=128,
        chunk_overlap=16,
    )


class TestReplSurvivesSearchErrors:
    def test_search_error_prints_failure_then_next_query_succeeds(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        project = _project(tmp_path)
        app_config = AppConfig()
        calls: list[str] = []

        class RecoveringEngine:
            def __init__(self, runtime: object) -> None:
                del runtime

            def search_project(self, project: ProjectConfig, config: AppConfig, options: object, freshness: object) -> object:
                del project, config, freshness
                calls.append(options.query)  # type: ignore[attr-defined]
                if len(calls) == 1:
                    raise SearchError("Last indexing run failed")

                class Outcome:
                    results: tuple[()] = ()

                return Outcome()

        queries = iter(["boom", "fine", ""])
        monkeypatch.setattr("typer.prompt", lambda *args, **kwargs: next(queries))
        monkeypatch.setattr(main_module, "load_app_config", lambda: app_config)
        monkeypatch.setattr(main_module, "require_active_project", lambda: project)
        monkeypatch.setattr(main_module, "_runtime_for_project", lambda factory, proj, config: object())
        monkeypatch.setattr(main_module, "SearchEngine", RecoveringEngine)

        main_module.repl()

        captured = capsys.readouterr()
        assert "Error: Last indexing run failed" in captured.err
        assert calls == ["boom", "fine"]


class TestResetRemovesDuckDBSidecars:
    def test_reset_removes_wal_and_tmp_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        project = _project(tmp_path)
        monkeypatch.setattr(main_module, "require_active_project", lambda: project)

        store = Store.open(project.metadata_db_path)
        store.close()
        wal_path = Path(str(project.metadata_db_path) + ".wal")
        wal_path.write_text("WAL")
        tmp_spill_dir = Path(str(project.metadata_db_path) + ".tmp")
        tmp_spill_dir.mkdir()
        (tmp_spill_dir / "spill.block").write_text("x")
        project.vector_index_path.write_bytes(b"idx")
        project.index_lock_path.write_text("lock")

        main_module.reset(yes=True)

        assert not wal_path.exists()
        assert not tmp_spill_dir.exists()
        assert not project.vector_index_path.exists()
        assert not project.index_lock_path.exists()

    def test_reset_removes_wal_and_tmp_sidecar_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        project = _project(tmp_path)
        monkeypatch.setattr(main_module, "require_active_project", lambda: project)

        Store.open(project.metadata_db_path).close()
        wal_path = Path(str(project.metadata_db_path) + ".wal")
        wal_path.write_text("WAL")
        tmp_spill_file = Path(str(project.metadata_db_path) + ".tmp")
        tmp_spill_file.write_text("partial spill")
        project.vector_index_path.write_bytes(b"idx")

        main_module.reset(yes=True)

        assert not wal_path.exists()
        assert not tmp_spill_file.exists()
        assert not project.metadata_db_path.exists()

    def test_reset_missing_sidecars_is_noop_and_fresh_store_is_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        project = _project(tmp_path)
        monkeypatch.setattr(main_module, "require_active_project", lambda: project)

        store = Store.open(project.metadata_db_path)
        store.close()

        main_module.reset(yes=True)

        assert not project.metadata_db_path.exists()
        fresh = Store.open(project.metadata_db_path)
        try:
            counts = fresh.counts(project.name)
            assert counts.files_count == 0
            assert counts.chunks_count == 0
        finally:
            fresh.close()
