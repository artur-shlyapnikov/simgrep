from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import simgrep.main as main_module
from simgrep.indexing import IndexEngine
from simgrep.models import SCHEMA_VERSION, AppConfig, IndexOptions, ProjectConfig


class _NoIndexRuntime:
    """Runtime stub whose vector-index factory must never be called by a dry-run."""

    def __init__(self) -> None:
        self.embedder_ndim = 4

    @property
    def embedder(self) -> Any:  # pragma: no cover - only reached if the bug regresses
        raise AssertionError("dry-run must not touch embedder")

    def new_vector_index(self, ndim: int) -> Any:
        raise AssertionError("dry-run must not construct a vector index")


def _project(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path,), "fake", 128, 20)


def _seed_project(tmp_path: Path, runtime: _NoIndexRuntime) -> ProjectConfig:
    (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
    # Seed a real index once via the normal (non-dry-run) path with a permissive stub.
    from tests.conftest import FakeRuntime

    real = FakeRuntime()
    seeded = _project(tmp_path)
    stats = IndexEngine(real).index_project(seeded, AppConfig(model="fake"), IndexOptions())
    assert stats.files_indexed == 1
    return seeded


class TestDryRunSkipsRuntime:
    def test_cli_dry_run_never_constructs_runtime(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
        runner = CliRunner()

        def boom(*args: object, **kwargs: object) -> None:
            raise AssertionError("dry-run must not construct Runtime")

        (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        monkeypatch.setattr(main_module, "_runtime_for_project", boom)
        # init creates the active project config without touching the runtime.
        result = runner.invoke(main_module.app, ["init", str(tmp_path)], catch_exceptions=False)
        assert result.exit_code == 0, result.output

        result = runner.invoke(main_module.app, ["index", "--dry-run"], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert "Would index 1 file(s)." in result.output

    def test_engine_dry_run_leaves_artifact_and_skips_index(self, tmp_path: Path) -> None:
        from tests.conftest import FakeRuntime

        (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
        project = _project(tmp_path)
        stats = IndexEngine(FakeRuntime()).index_project(project, AppConfig(model="fake"), IndexOptions())
        assert stats.files_indexed == 1
        artifact_before = project.vector_index_path.read_bytes()
        (tmp_path / "b.py").write_text("bravo charlie", encoding="utf-8")

        engine = IndexEngine(_NoIndexRuntime())
        dry_stats = engine.index_project(project, AppConfig(model="fake"), IndexOptions(dry_run=True))
        assert dry_stats.files_seen == 2
        assert dry_stats.files_indexed == 1
        assert dry_stats.files_skipped_unchanged == 1
        assert project.vector_index_path.read_bytes() == artifact_before
