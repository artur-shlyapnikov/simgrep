from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simgrep.errors import SearchError
from simgrep.indexing import IndexEngine
from simgrep.models import (
    SCHEMA_VERSION,
    AppConfig,
    FreshnessMode,
    IndexOptions,
    IndexState,
    ProjectConfig,
    SearchOptions,
)
from simgrep.search import SearchEngine
from simgrep.store import Store
from tests.conftest import FakeEmbedder, FakeRuntime


def _project(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path,), "fake", 128, 20)


def _indexed_project(tmp_path: Path, fake_runtime: FakeRuntime) -> ProjectConfig:
    (tmp_path / "a.py").write_text("alpha beta gamma", encoding="utf-8")
    project = _project(tmp_path)
    IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))
    return project


def _simulate_crashed_indexer(project: ProjectConfig) -> None:
    store = Store.open(project.metadata_db_path)
    try:
        store.set_meta("index_state", IndexState.indexing.value)
    finally:
        store.close()


@pytest.mark.parametrize("freshness", [FreshnessMode.auto, FreshnessMode.skip, FreshnessMode.check])
def test_stale_indexing_flag_recovers_instead_of_raising(tmp_path: Path, fake_runtime: FakeRuntime, freshness: FreshnessMode) -> None:
    project = _indexed_project(tmp_path, fake_runtime)
    app_config = AppConfig(model="fake")
    engine = SearchEngine(fake_runtime)

    # RED: with a stale "indexing" flag the search currently fails permanently.
    # RED evidence (pre-fix): this call raised SearchError("Index is currently being built.").
    _simulate_crashed_indexer(project)

    # GREEN: recovery rebuilds/repairs and the search succeeds.
    outcome = engine.search_project(project, app_config, SearchOptions(query="alpha"), freshness)
    assert len(outcome.results) > 0

    store = Store.open(project.metadata_db_path)
    try:
        assert store.get_meta("index_state") == IndexState.ready.value
        chunks_count = store.counts(project.name).chunks_count
    finally:
        store.close()

    # A second search succeeds without further indexing work.
    second = engine.search_project(project, app_config, SearchOptions(query="alpha"), freshness)
    assert len(second.results) > 0
    store = Store.open(project.metadata_db_path)
    try:
        assert store.get_meta("index_state") == IndexState.ready.value
        assert store.counts(project.name).chunks_count == chunks_count
    finally:
        store.close()


def test_healed_search_includes_files_created_while_crashed(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    """Self-heal runs a real incremental reindex, so files added during the 'crash' become searchable."""
    project = _indexed_project(tmp_path, fake_runtime)
    (tmp_path / "b.py").write_text("delta epsilon zeta", encoding="utf-8")
    app_config = AppConfig(model="fake")
    engine = SearchEngine(fake_runtime)
    _simulate_crashed_indexer(project)

    outcome = engine.search_project(project, app_config, SearchOptions(query="delta"), FreshnessMode.skip)

    assert "b.py" in [result.file_path.name for result in outcome.results]
    assert outcome.chunks_searched == 2
    store = Store.open(project.metadata_db_path)
    try:
        assert store.get_meta("index_state") == IndexState.ready.value
    finally:
        store.close()


class ExplodingEmbedder(FakeEmbedder):
    """Fails while the crashed indexer would have been embedding chunks."""

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int | None = None) -> np.ndarray:
        del texts, is_query, batch_size
        raise RuntimeError("embedder exploded during heal")


def test_builder_failure_during_self_heal_propagates_and_marks_failed(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    """If repair fails, the builder error surfaces unwrapped, the store flips to failed,
    and the next search refuses with the clean failed-state error."""
    project = _indexed_project(tmp_path, fake_runtime)
    (tmp_path / "b.py").write_text("delta epsilon zeta", encoding="utf-8")  # pending mutation forces encode() during heal
    app_config = AppConfig(model="fake")
    engine = SearchEngine(fake_runtime)
    _simulate_crashed_indexer(project)

    healthy_embedder = fake_runtime.embedder
    fake_runtime.embedder = ExplodingEmbedder()
    with pytest.raises(RuntimeError, match="embedder exploded"):
        engine.search_project(project, app_config, SearchOptions(query="alpha"), FreshnessMode.skip)

    store = Store.open(project.metadata_db_path)
    try:
        assert store.get_meta("index_state") == IndexState.failed.value
        assert store.get_meta("last_index_error") == "embedder exploded during heal"
    finally:
        store.close()

    fake_runtime.embedder = healthy_embedder
    with pytest.raises(SearchError, match="Last indexing run failed") as exc_info:
        engine.search_project(project, app_config, SearchOptions(query="alpha"), FreshnessMode.skip)
    assert exc_info.value.hint == "embedder exploded during heal"


def test_stale_flag_with_missing_vector_index_still_refuses(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    """Self-heal covers a lingering flag only; a missing vector artifact still refuses cleanly."""
    project = _indexed_project(tmp_path, fake_runtime)
    project.vector_index_path.unlink()
    app_config = AppConfig(model="fake")
    _simulate_crashed_indexer(project)

    with pytest.raises(SearchError, match="Vector index not found"):
        SearchEngine(fake_runtime).search_project(project, app_config, SearchOptions(query="alpha"), FreshnessMode.skip)
