from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pytest

from simgrep.indexing import IndexEngine
from simgrep.models import (
    SCHEMA_VERSION,
    AppConfig,
    Chunk,
    EphemeralIndexOptions,
    IndexOptions,
    IndexStats,
    ProjectConfig,
)
from simgrep.store import Store
from tests.conftest import FakeRuntime


def _project(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path,), "fake", 128, 20)


class _ExplodingIndex:
    """Delegates everything to a real index but makes index.add fail."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __len__(self) -> int:
        return len(self._inner)

    def add(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("simulated vector index write failure")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _ExplodingIndexRuntime(FakeRuntime):
    def new_vector_index(self, ndim: int) -> Any:
        return _ExplodingIndex(super().new_vector_index(ndim))


class WordChunker:
    """One chunk per word, so chunk count per file equals its word count."""

    def chunk(self, text: str) -> Sequence[Chunk]:
        if not text.strip():
            return []
        out: list[Chunk] = []
        pos = 0
        for word in text.split():
            idx = text.index(word, pos)
            out.append(Chunk(id=-1, file_id=-1, text=word, start=idx, end=idx + len(word), tokens=1))
            pos = idx + len(word)
        return out


class _SpyIndex:
    def __init__(self, inner: Any, runtime: "_FlushSpyRuntime") -> None:
        self._inner = inner
        self._runtime = runtime

    def __len__(self) -> int:
        return len(self._inner)

    def add(self, labels: Any = None, vectors: Any = None, **kwargs: Any) -> None:
        self._runtime.add_sizes.append(int(len(labels)))
        self._inner.add(labels, vectors, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _CountingEmbedder:
    """Records encode batch sizes while delegating to the real fake embedder."""

    def __init__(self, inner: Any, sizes: list[int]) -> None:
        self._inner = inner
        self._sizes = sizes

    @property
    def ndim(self) -> int:
        return int(self._inner.ndim)

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int | None = None) -> Any:
        self._sizes.append(len(texts))
        return self._inner.encode(texts, is_query=is_query, batch_size=batch_size)


class _FlushSpyRuntime(FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.chunker: Any = WordChunker()
        self.encode_sizes: list[int] = []
        self.add_sizes: list[int] = []
        self.embedder: Any = _CountingEmbedder(self.embedder, self.encode_sizes)

    def new_vector_index(self, ndim: int) -> Any:
        return _SpyIndex(super().new_vector_index(ndim), self)


def test_noop_rerun_skips_term_stats_refresh_and_vector_rewrite(tmp_path: Path, fake_runtime: FakeRuntime, monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-indexing an unchanged corpus must be side-effect free: term stats are
    not refreshed and the on-disk vector file is not rewritten (save-gate false arm)."""
    (tmp_path / "a.py").write_text("alpha beta gamma", encoding="utf-8")
    project = _project(tmp_path)
    IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

    vpath = project.vector_index_path
    stat_before = vpath.stat()

    refresh_calls: list[int] = []
    original_refresh = Store.refresh_term_stats

    def _spy_refresh(self: Store) -> None:
        refresh_calls.append(1)
        original_refresh(self)

    monkeypatch.setattr(Store, "refresh_term_stats", _spy_refresh)
    stats = IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions())

    assert stats.index_mutated is False
    assert stats.files_skipped_unchanged == 1
    assert refresh_calls == [], "noop rerun must not refresh term stats"
    stat_after = vpath.stat()
    assert stat_after.st_ino == stat_before.st_ino, "noop rerun must not rewrite the vector file"
    assert stat_after.st_mtime_ns == stat_before.st_mtime_ns


def test_rebuild_of_empty_corpus_saves_empty_index_and_resets_store(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    """Save-gate row (rebuild=True, no mutations): rebuilding an empty corpus still
    atomically saves an empty vector index and leaves index_state ready."""
    project = _project(tmp_path)
    stats = IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

    assert stats.files_seen == 0
    assert stats.files_processed == 0
    assert stats.chunks_indexed == 0
    assert project.vector_index_path.exists(), "rebuild must save the index even with zero mutations"
    assert project.vector_index_path.read_text(encoding="utf-8") == "", "empty corpus must save an empty vector index"

    store = Store.open(project.metadata_db_path)
    try:
        assert store.get_meta("index_state") == "ready"
        assert store.get_files() == {}
    finally:
        store.close()


def test_missing_vector_file_is_saved_on_zero_mutation_run(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    """Save-gate third disjunct (mutated=False, rebuild=False, vector file missing): with a
    zero-mutation corpus the engine still saves an (empty) index so search never finds a missing vector file.
    NOTE: deleting the vector file over a POPULATED store instead triggers the divergence
    rebuild (unloaded index has len 0 < chunks_count), so that route must NOT be used here."""
    (tmp_path / "ws.py").write_text("   \n\t\n  \n", encoding="utf-8")
    project = _project(tmp_path)
    stats = IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions())

    assert stats.index_mutated is False
    assert stats.files_indexed == 1, "chunk-less file is recorded (no chunks, no vectors)"
    assert project.vector_index_path.exists(), "missing vector file must be created even with zero mutations"
    assert project.vector_index_path.read_text(encoding="utf-8") == "", "empty store must persist as an empty vector index"

    store = Store.open(project.metadata_db_path)
    try:
        assert store.get_meta("index_state") == "ready"
    finally:
        store.close()


def test_compensation_delete_failure_does_not_mask_original_error(tmp_path: Path, fake_runtime: FakeRuntime, monkeypatch: pytest.MonkeyPatch) -> None:
    """If store.delete_file raises while rolling back inserted rows, the swallow-guard at
    indexing.py:140-143 must let the ORIGINAL index failure propagate, not the compensation error."""
    (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
    (tmp_path / "b.py").write_text("gamma delta", encoding="utf-8")
    project = _project(tmp_path)

    def _exploding_delete(self: Store, file_id: int) -> list[int]:
        raise RuntimeError("compensation boom")

    monkeypatch.setattr(Store, "delete_file", _exploding_delete)

    with pytest.raises(RuntimeError, match="simulated vector index write failure") as excinfo:
        IndexEngine(_ExplodingIndexRuntime()).index_project(project, AppConfig(model="fake"), IndexOptions())
    assert "compensation boom" not in str(excinfo.value), "compensation failure must never mask the original error"

    store = Store.open(project.metadata_db_path)
    try:
        assert store.get_meta("index_state") == "failed"
        assert "simulated vector index write failure" in (store.get_meta("last_index_error") or "")
    finally:
        store.close()


def test_successful_rerun_after_failure_clears_last_index_error(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    """State transition: a failed run records index_state=failed plus last_index_error;
    the next successful run flips to ready and wipes last_index_error."""
    (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
    (tmp_path / "b.py").write_text("gamma delta", encoding="utf-8")
    project = _project(tmp_path)

    with pytest.raises(RuntimeError, match="simulated vector index write failure"):
        IndexEngine(_ExplodingIndexRuntime()).index_project(project, AppConfig(model="fake"), IndexOptions())

    store = Store.open(project.metadata_db_path)
    try:
        assert store.get_meta("index_state") == "failed"
        assert "simulated vector index write failure" in (store.get_meta("last_index_error") or "")
    finally:
        store.close()

    stats = IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions())
    assert stats.files_skipped_unchanged == 0, "compensated files must be planned as new again"
    assert stats.files_indexed == 2

    store = Store.open(project.metadata_db_path)
    try:
        assert store.get_meta("index_state") == "ready"
        assert store.get_meta("last_index_error") == "", "successful run must clear last_index_error"
    finally:
        store.close()


def test_build_ephemeral_closes_store_when_flush_fails(tmp_path: Path, fake_runtime: FakeRuntime, monkeypatch: pytest.MonkeyPatch) -> None:
    """When embedding fails inside build_ephemeral, the except-close guard (lines 170-172)
    must close the in-memory store exactly once and re-raise the original error."""
    (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")

    def _boom(*args: object, **kwargs: object) -> Any:
        raise RuntimeError("encode exploded")

    monkeypatch.setattr(fake_runtime.embedder, "encode", _boom)

    closed: list[int] = []
    original_close = Store.close

    def _spy_close(self: Store) -> None:
        closed.append(1)
        original_close(self)

    monkeypatch.setattr(Store, "close", _spy_close)

    with pytest.raises(RuntimeError, match="encode exploded"):
        IndexEngine(fake_runtime).build_ephemeral([tmp_path / "a.py"], AppConfig(model="fake"), EphemeralIndexOptions())

    assert closed == [1], f"store must be closed exactly once on failure, got {closed}"


def test_whitespace_only_file_yields_zero_chunks_but_is_recorded(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    """Empty-chunk boundary: a whitespace-only file produces zero chunks from the chunker.
    It must be recorded as a file row (with no chunk rows and no vectors) so later
    freshness plans report it as unchanged instead of re-reporting it as new forever."""
    (tmp_path / "ws.py").write_text("   \n\t\n  \n", encoding="utf-8")
    project = _project(tmp_path)
    stats = IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

    assert stats.files_seen == 1
    assert stats.files_processed == 1
    assert stats.files_indexed == 1, "chunk-less file is recorded so plans stop re-reporting it"
    assert stats.chunks_indexed == 0
    assert stats.index_mutated is False

    store = Store.open(project.metadata_db_path)
    try:
        assert store.counts(project.name).chunks_count == 0
        files = store.get_files()
        assert len(files) == 1, "chunk-less file gets a file row (no chunks, no vectors)"
    finally:
        store.close()
    assert project.vector_index_path.exists()  # rebuild=True forces the save despite zero mutations
    assert project.vector_index_path.read_text(encoding="utf-8") == ""

    # Freshness regression guard: the second plan must report no mutations.
    follow_up = IndexEngine(fake_runtime).plan_project(project, AppConfig(model="fake"), IndexOptions())
    assert follow_up.has_mutations is False


def test_mid_run_flush_splits_batches_at_threshold(tmp_path: Path) -> None:
    """Flush-threshold boundary: with one chunk per word and word counts 100/100/100/20,
    pending crosses max(batch_size*4, 256)=256 mid-loop (flush of 300) and the trailing
    20 chunks go through the final flush — proving both flush sites fire."""
    for name, words in (("a.py", 100), ("b.py", 100), ("c.py", 100), ("d.py", 20)):
        (tmp_path / name).write_text(" ".join(f"w{i}q{i}" for i in range(words)), encoding="utf-8")
    project = _project(tmp_path)
    runtime = _FlushSpyRuntime()

    stats = IndexEngine(runtime).index_project(project, AppConfig(model="fake", batch_size=32), IndexOptions(rebuild=True))

    assert runtime.encode_sizes == [300, 20], f"expected mid-run flush at 300 >= 256 plus final flush of 20, got {runtime.encode_sizes}"
    assert runtime.add_sizes == [300, 20]
    assert stats.chunks_indexed == 320
    assert stats.vectors_added == 320

    store = Store.open(project.metadata_db_path)
    try:
        assert store.counts(project.name).chunks_count == 320
    finally:
        store.close()
    saved_labels = [x for x in project.vector_index_path.read_text(encoding="utf-8").split(",") if x]
    assert len(saved_labels) == 320


def test_insert_prepared_with_zero_chunks_is_a_noop_guard(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    """Defensive guard (lines 270-272): inserting an empty prepared batch must reserve no
    labels, touch no store rows, and leave stats untouched. Direct private-call contract test."""
    engine = IndexEngine(fake_runtime)
    store = Store.memory()
    try:
        inserted_file_ids: list[int] = []
        stats = IndexStats()
        engine._insert_prepared([], store, fake_runtime.new_vector_index(4), AppConfig(model="fake"), stats, inserted_file_ids)
        assert inserted_file_ids == []
        assert stats.chunks_indexed == 0
        assert stats.vectors_added == 0
        assert stats.index_mutated is False
        assert store.counts("p").chunks_count == 0
    finally:
        store.close()
