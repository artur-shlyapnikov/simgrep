from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from simgrep.indexing import IndexEngine
from simgrep.models import SCHEMA_VERSION, AppConfig, IndexOptions, ProjectConfig
from simgrep.store import Store
from tests.conftest import FakeEmbedder, FakeRuntime


def _project(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path,), "fake", 128, 20)


def test_index_project_dry_run_only_plans(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    (tmp_path / "a.py").write_text("print(1)", encoding="utf-8")
    project = _project(tmp_path)
    stats = IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(dry_run=True))
    assert stats.files_seen == 1
    store = Store.open(project.metadata_db_path)
    try:
        assert store.counts().chunks_count == 0
    finally:
        store.close()


def test_index_project_rebuild_indexes_and_saves_vector(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    (tmp_path / "a.py").write_text("rollback payment", encoding="utf-8")
    project = _project(tmp_path)
    stats = IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))
    assert stats.files_indexed == 1
    assert stats.chunks_indexed == 1
    assert project.vector_index_path.exists()


def test_index_project_atomic_save_keeps_old_index_on_failure(tmp_path: Path) -> None:
    class FailingSaveIndex:
        def __init__(self, ndim: int = 4) -> None:
            self.ndim = ndim
            self.labels: list[int] = []

        def __len__(self) -> int:
            return len(self.labels)

        def add(self, labels, vectors) -> None:  # type: ignore[no-untyped-def]
            del vectors
            self.labels.extend(int(label) for label in labels)

        def remove(self, labels) -> None:  # type: ignore[no-untyped-def]
            for label in labels:
                value = int(label)
                if value in self.labels:
                    self.labels.remove(value)

        def save(self, path: Path) -> None:
            if path.suffix == ".tmp":
                raise RuntimeError("boom")
            path.write_text("stable", encoding="utf-8")

        def load(self, path: Path) -> None:
            del path

    class RuntimeWithFailingSave:
        def __init__(self) -> None:
            base = FakeRuntime()
            self.extractor = base.extractor
            self.chunker = base.chunker
            self.embedder = base.embedder

        def new_vector_index(self, ndim: int) -> Any:
            return FailingSaveIndex(ndim)

    (tmp_path / "a.py").write_text("first", encoding="utf-8")
    project = _project(tmp_path)
    project.vector_index_path.parent.mkdir(parents=True, exist_ok=True)
    project.vector_index_path.write_text("stable", encoding="utf-8")

    with pytest.raises(RuntimeError, match="boom"):
        IndexEngine(RuntimeWithFailingSave()).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

    assert project.vector_index_path.read_text(encoding="utf-8") == "stable"
    store = Store.open(project.metadata_db_path)
    try:
        assert store.get_meta("index_state") == "failed"
    finally:
        store.close()


def test_build_ephemeral_single_file_base_and_oserror_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_runtime: FakeRuntime) -> None:
    target = tmp_path / "solo.py"
    target.write_text("quokka marker content\n", encoding="utf-8")

    def oserror_hash(path: Path) -> str:
        raise OSError("unreadable")

    monkeypatch.setattr("simgrep.indexing.calculate_file_hash", oserror_hash)
    handle = IndexEngine(fake_runtime).build_ephemeral([target], AppConfig(model="fake"))
    try:
        assert handle.base_path == tmp_path
        # Infrastructure probe: hash policy lives under the corpus boundary.
        record = handle._store.get_files()[target.resolve()]
        assert record.sha256 is None
    finally:
        handle.close()


class _ExplodingEmbedder(FakeEmbedder):
    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int | None = None) -> np.ndarray:
        raise RuntimeError("embed boom")


class _ExplodingRuntime(FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.embedder = _ExplodingEmbedder()


def test_build_ephemeral_embedder_failure_closes_store_and_reraises(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("some content\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="embed boom"):
        IndexEngine(_ExplodingRuntime()).build_ephemeral([tmp_path], AppConfig(model="fake"))
