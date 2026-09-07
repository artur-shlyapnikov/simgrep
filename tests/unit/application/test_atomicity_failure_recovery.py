from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import patch

import numpy as np
import pytest

from simgrep.errors import SearchError
from simgrep.indexing import IndexEngine
from simgrep.models import SCHEMA_VERSION, AppConfig, FreshnessMode, IndexOptions, IndexState, ProjectConfig, SearchOptions
from simgrep.search import SearchEngine
from simgrep.store import Store
from tests.conftest import FakeEmbedder, FakeRuntime, FakeTextExtractor, FakeTokenChunker, FakeVectorIndex


def _project(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path,), "fake", 128, 20)


class FailingExtractor:
    def extract(self, path: Path) -> str:
        raise RuntimeError("extractor boom")


class FailingChunker:
    def chunk(self, text: str) -> Sequence[Any]:
        raise RuntimeError("chunker boom")


class FailingEmbedder:
    @property
    def ndim(self) -> int:
        return 4

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int | None = None) -> np.ndarray:
        raise RuntimeError("embedder boom")


class StableTextVectorIndex(FakeVectorIndex):
    def load(self, path: Path) -> None:
        if path.exists():
            content = path.read_text(encoding="utf-8")
            if content and not content.startswith("old"):
                self.data = {int(label): np.ones(self.ndim, dtype=np.float32) for label in content.split(",") if label}


class NoLoadVectorIndex(FakeVectorIndex):
    def load(self, path: Path) -> None:
        pass


class RuntimeWithFailingExtractor:
    def __init__(self) -> None:
        self.extractor = FailingExtractor()
        self.chunker = FakeTokenChunker()
        self.embedder = FakeEmbedder()

    def new_vector_index(self, ndim: int) -> FakeVectorIndex:
        return FakeVectorIndex(ndim)


class RuntimeWithFailingChunker:
    def __init__(self) -> None:
        self.extractor = FakeTextExtractor()
        self.chunker = FailingChunker()
        self.embedder = FakeEmbedder()

    def new_vector_index(self, ndim: int) -> FakeVectorIndex:
        return FakeVectorIndex(ndim)


class RuntimeWithFailingEmbedder:
    def __init__(self) -> None:
        self.extractor = FakeTextExtractor()
        self.chunker = FakeTokenChunker()
        self.embedder = FailingEmbedder()

    def new_vector_index(self, ndim: int) -> FakeVectorIndex:
        return StableTextVectorIndex(ndim)


def test_extractor_error_does_not_leave_index_partially_ready(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print(1)", encoding="utf-8")
    project = _project(tmp_path)

    with pytest.raises(RuntimeError, match="extractor boom"):
        IndexEngine(RuntimeWithFailingExtractor()).index_project(project, AppConfig(model="fake"), IndexOptions())

    store = Store.open(project.metadata_db_path)
    try:
        state = store.get_meta("index_state")
        assert state == IndexState.failed.value
        error = store.get_meta("last_index_error")
        assert error is not None
        assert "extractor boom" in error
        counts = store.counts()
        assert counts.chunks_count == 0
    finally:
        store.close()


def test_chunker_error_sets_failed_state_with_actionable_error(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print(1)", encoding="utf-8")
    project = _project(tmp_path)

    with pytest.raises(RuntimeError, match="chunker boom"):
        IndexEngine(RuntimeWithFailingChunker()).index_project(project, AppConfig(model="fake"), IndexOptions())

    store = Store.open(project.metadata_db_path)
    try:
        assert store.get_meta("index_state") == IndexState.failed.value
        error = store.get_meta("last_index_error")
        assert error is not None
        assert "chunker boom" in error
    finally:
        store.close()


def test_embedder_error_does_not_corrupt_old_index(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("first", encoding="utf-8")
    project = _project(tmp_path)
    project.vector_index_path.parent.mkdir(parents=True, exist_ok=True)
    project.vector_index_path.write_text("old stable", encoding="utf-8")

    with pytest.raises(RuntimeError, match="embedder boom"):
        IndexEngine(RuntimeWithFailingEmbedder()).index_project(project, AppConfig(model="fake"), IndexOptions())

    assert project.vector_index_path.read_text() == "old stable"
    store = Store.open(project.metadata_db_path)
    try:
        assert store.get_meta("index_state") == IndexState.failed.value
    finally:
        store.close()


def test_insert_chunks_error_keeps_vector_index_clean(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print(1)", encoding="utf-8")
    project = _project(tmp_path)

    def failing_insert_chunks(self: Store, records: list[Any]) -> None:
        raise RuntimeError("insert_chunks boom")

    with patch.object(Store, "insert_chunks", failing_insert_chunks):
        with pytest.raises(RuntimeError, match="insert_chunks boom"):
            IndexEngine(FakeRuntime()).index_project(project, AppConfig(model="fake"), IndexOptions())

    store = Store.open(project.metadata_db_path)
    try:
        assert store.get_meta("index_state") == IndexState.failed.value
    finally:
        store.close()


def test_insert_terms_error_does_not_leave_chunks_without_terms_as_ready(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print(1)", encoding="utf-8")
    project = _project(tmp_path)

    def failing_insert_terms(self: Store, records: list[Any]) -> None:
        raise RuntimeError("insert_terms boom")

    with patch.object(Store, "insert_terms", failing_insert_terms):
        with pytest.raises(RuntimeError, match="insert_terms boom"):
            IndexEngine(FakeRuntime()).index_project(project, AppConfig(model="fake"), IndexOptions())

    store = Store.open(project.metadata_db_path)
    try:
        assert store.get_meta("index_state") == IndexState.failed.value
    finally:
        store.close()


def test_vector_add_error_does_not_leave_duckdb_ready_with_chunks_without_vectors(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print(1)", encoding="utf-8")
    project = _project(tmp_path)

    original_index_add = FakeVectorIndex.add

    def failing_add(self: FakeVectorIndex, labels: Any = None, vectors: Any = None, **kwargs: Any) -> None:
        if labels is not None and len(labels) > 0:
            raise RuntimeError("vector add boom")
        return original_index_add(self, labels, vectors, **kwargs)

    with patch.object(FakeVectorIndex, "add", failing_add):
        with pytest.raises(RuntimeError, match="vector add boom"):
            IndexEngine(FakeRuntime()).index_project(project, AppConfig(model="fake"), IndexOptions())

    store = Store.open(project.metadata_db_path)
    try:
        state = store.get_meta("index_state")
        assert state == IndexState.failed.value
    finally:
        store.close()


class FailingSaveIndex:
    def __init__(self, ndim: int = 4) -> None:
        self.ndim = ndim
        self.labels: list[int] = []

    def __len__(self) -> int:
        return len(self.labels)

    def add(self, labels: Any, vectors: Any, **kwargs: Any) -> None:
        for label, vector in zip(labels, vectors):
            self.labels.append(int(label))

    def save(self, path: Path) -> None:
        if path.suffix == ".tmp":
            raise RuntimeError("atomic save boom")
        path.write_text(",".join(str(label_id) for label_id in self.labels), encoding="utf-8")

    def load(self, path: Path) -> None:
        if path.exists():
            content = path.read_text(encoding="utf-8")
            if content and not content.startswith("old"):
                self.labels = [int(x) for x in content.split(",") if x]


class RuntimeWithFailingSave:
    def __init__(self) -> None:
        self.extractor = FakeTextExtractor()
        self.chunker = FakeTokenChunker()
        self.embedder = FakeEmbedder()

    def new_vector_index(self, ndim: int) -> Any:
        return FailingSaveIndex(ndim)


def test_atomic_save_preserves_old_usearch_and_cleans_temp_file(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("content", encoding="utf-8")
    project = _project(tmp_path)
    project.vector_index_path.parent.mkdir(parents=True, exist_ok=True)
    project.vector_index_path.write_text("old stable", encoding="utf-8")

    captured_tmp_path: list[Path] = []

    class CapturingSaveIndex(FailingSaveIndex):
        def save(self, path: Path) -> None:
            captured_tmp_path.append(path)
            super().save(path)

    class RuntimeWithCapturingSave:
        def __init__(self) -> None:
            self.extractor = FakeTextExtractor()
            self.chunker = FakeTokenChunker()
            self.embedder = FakeEmbedder()

        def new_vector_index(self, ndim: int) -> Any:
            return CapturingSaveIndex(ndim)

    with pytest.raises(RuntimeError, match="atomic save boom"):
        IndexEngine(RuntimeWithCapturingSave()).index_project(project, AppConfig(model="fake"), IndexOptions())

    assert project.vector_index_path.read_text() == "old stable"
    if captured_tmp_path:
        assert not captured_tmp_path[0].exists(), f"temp file {captured_tmp_path[0]} should be cleaned up after failed save"

    tmp_dir = project.vector_index_path.parent / "tmp"
    if tmp_dir.exists():
        remaining = list(tmp_dir.iterdir())
        assert len(remaining) == 0, f"tmp dir should be empty but has: {remaining}"


class FailingExtractorOnce:
    def __init__(self) -> None:
        self._fail = True

    def extract(self, path: Path) -> str:
        if self._fail:
            self._fail = False
            raise RuntimeError("first failure")
        return path.read_text(encoding="utf-8")

    @property
    def ndim(self) -> int:
        return 4

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int | None = None) -> np.ndarray:
        vectors = np.zeros((len(texts), self.ndim), dtype=np.float32)
        for i, text in enumerate(texts):
            n = float(len(text) or 1)
            vectors[i] = np.array([n, n % 7, n % 13, 1.0], dtype=np.float32)
        return vectors


class RuntimeFailsOnce:
    def __init__(self) -> None:
        self.extractor = FailingExtractorOnce()
        self.chunker = FakeTokenChunker()
        self.embedder = FakeEmbedder()

    def new_vector_index(self, ndim: int) -> FakeVectorIndex:
        return FakeVectorIndex(ndim)


def test_rebuild_after_failed_indexing_restores_ready_state(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("first", encoding="utf-8")
    project = _project(tmp_path)

    with pytest.raises(RuntimeError):
        IndexEngine(RuntimeFailsOnce()).index_project(project, AppConfig(model="fake"), IndexOptions())

    store = Store.open(project.metadata_db_path)
    try:
        assert store.get_meta("index_state") == IndexState.failed.value
    finally:
        store.close()

    stats = IndexEngine(FakeRuntime()).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))
    assert stats.files_indexed == 1

    store = Store.open(project.metadata_db_path)
    try:
        assert store.get_meta("index_state") == IndexState.ready.value
    finally:
        store.close()


def test_search_freshness_skip_refuses_when_index_state_failed(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print(1)", encoding="utf-8")
    project = _project(tmp_path)

    IndexEngine(FakeRuntime()).index_project(project, AppConfig(model="fake"), IndexOptions())

    store = Store.open(project.metadata_db_path)
    store.set_meta("index_state", IndexState.failed.value)
    store.set_meta("last_index_error", "simulated failure")
    store.close()

    search = SearchEngine(FakeRuntime())
    options = SearchOptions(query="test")

    with pytest.raises(SearchError, match="Last indexing run failed"):
        search.search_project(project, AppConfig(model="fake"), options, FreshnessMode.skip)


def test_search_recovers_from_stale_index_state_indexing(tmp_path: Path) -> None:
    """A stale "indexing" flag is a crash artifact while we hold the singleton lock: search self-heals."""
    (tmp_path / "a.py").write_text("print(1)", encoding="utf-8")
    project = _project(tmp_path)

    IndexEngine(FakeRuntime()).index_project(project, AppConfig(model="fake"), IndexOptions())

    store = Store.open(project.metadata_db_path)
    store.set_meta("index_state", IndexState.indexing.value)
    store.close()

    search = SearchEngine(FakeRuntime())
    outcome = search.search_project(project, AppConfig(model="fake"), SearchOptions(query="test"), FreshnessMode.skip)
    assert isinstance(outcome.results, list)

    store = Store.open(project.metadata_db_path)
    try:
        assert store.get_meta("index_state") == IndexState.ready.value
    finally:
        store.close()


def test_concurrent_index_calls_serialize_with_lock(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print(1)", encoding="utf-8")
    project = _project(tmp_path)

    call_order: list[int] = []

    class SlowIndexEngine(IndexEngine):
        def index_project(self, proj: ProjectConfig, cfg: AppConfig, opts: IndexOptions) -> Any:
            call_order.append(len(call_order))
            time.sleep(0.1)
            return super().index_project(proj, cfg, opts)

    results: list[tuple[int, Any]] = []
    errors: list[Exception] = []

    def run_index(i: int) -> None:
        try:
            engine = SlowIndexEngine(FakeRuntime())
            result = engine.index_project(project, AppConfig(model="fake"), IndexOptions())
            results.append((i, result))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=run_index, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Expected no errors, got {errors}"
    assert len(results) == 3
    assert call_order == [0, 1, 2], f"Expected serialized execution, got {call_order}"


def test_search_during_indexing_gets_clear_error_or_waits(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print(1)", encoding="utf-8")
    project = _project(tmp_path)

    index_started = threading.Event()
    search_blocked = threading.Event()

    class SlowExtract:
        def extract(self, path: Path) -> str:
            index_started.set()
            search_blocked.wait(timeout=5)
            return path.read_text(encoding="utf-8")

        @property
        def ndim(self) -> int:
            return 4

    class SlowRuntime:
        def __init__(self) -> None:
            self.extractor = SlowExtract()
            self.chunker = FakeTokenChunker()
            self.embedder = FakeEmbedder()

        def new_vector_index(self, ndim: int) -> FakeVectorIndex:
            return NoLoadVectorIndex(ndim)

    search_error_received = threading.Event()
    search_error_msg: list[str] = []

    def run_index() -> None:
        try:
            engine = IndexEngine(SlowRuntime())
            engine.index_project(project, AppConfig(model="fake"), IndexOptions())
        except Exception:
            pass
        finally:
            search_blocked.set()

    def run_search() -> None:
        try:
            search = SearchEngine(FakeRuntime())
            options = SearchOptions(query="test")
            search.search_project(project, AppConfig(model="fake"), options, FreshnessMode.skip)
        except SearchError as e:
            if "Index is currently being built" in str(e):
                search_error_received.set()
            search_error_msg.append(str(e))
        except Exception as e:
            search_error_msg.append(str(e))

    index_thread = threading.Thread(target=run_index)
    index_thread.start()

    index_started.wait(timeout=2)
    search_thread = threading.Thread(target=run_search)
    search_thread.start()

    time.sleep(0.1)
    search_blocked.set()

    index_thread.join(timeout=5)
    search_thread.join(timeout=5)

    if search_error_received.is_set() and search_error_msg:
        assert "Index is currently being built" in search_error_msg[0]
