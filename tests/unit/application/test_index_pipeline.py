"""Pipeline writer tests: encode/write overlap in IndexEngine._flush_prepared.

Covers the behaviors the serial path could not have: writer-thread failures
propagating with full compensation, encode failures rolling back committed
flushes, label determinism across runs, and a deterministic proof that flush
N's index.add completes while flush N+1's encode is still blocked.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Sequence

import pytest

from simgrep.indexing import IndexEngine
from simgrep.models import SCHEMA_VERSION, AppConfig, Chunk, IndexOptions, ProjectConfig
from simgrep.store import Store
from tests.conftest import FakeRuntime


def _project(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path,), "fake", 128, 20)


class WordChunker:
    """One chunk per word so flush boundaries are controllable via word counts."""

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


def _word_runtime(base: FakeRuntime) -> FakeRuntime:
    base.chunker = WordChunker()  # type: ignore[assignment]
    return base


class _OverlapGateEmbedder:
    """Delegates to the fake embedder; the Nth encode blocks until the writer has
    completed a flush's index.add. A serial (non-pipelined) flow times out here."""

    def __init__(self, inner: Any, block_on_call: int) -> None:
        self._inner = inner
        self._block_on_call = block_on_call
        self._calls = 0
        self.first_add_done = threading.Event()

    @property
    def ndim(self) -> int:
        return int(self._inner.ndim)

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int | None = None) -> Any:
        self._calls += 1
        if self._calls == self._block_on_call:
            assert self.first_add_done.wait(timeout=10.0), "writer did not overlap with encode"
        return self._inner.encode(texts, is_query=is_query, batch_size=batch_size)


class _SpyIndex:
    def __init__(self, inner: Any, gate: _OverlapGateEmbedder) -> None:
        self._inner = inner
        self._gate = gate

    def __len__(self) -> int:
        return len(self._inner)

    def add(self, labels: Any = None, vectors: Any = None, **kwargs: Any) -> None:
        self._inner.add(labels, vectors, **kwargs)
        self._gate.first_add_done.set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _ExplodingOnNthAddIndex:
    """Real index behavior, but the Nth add call raises."""

    def __init__(self, inner: Any, fail_on_call: int) -> None:
        self._inner = inner
        self._fail_on_call = fail_on_call
        self._calls = 0

    def __len__(self) -> int:
        return len(self._inner)

    def add(self, labels: Any = None, vectors: Any = None, **kwargs: Any) -> None:
        self._calls += 1
        if self._calls == self._fail_on_call:
            raise RuntimeError("simulated vector index write failure")
        self._inner.add(labels, vectors, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _write_words(tmp_path: Path, names: tuple[str, ...], words: int) -> None:
    for name in names:
        (tmp_path / name).write_text(" ".join(f"w{i}q{i}" for i in range(words)), encoding="utf-8")


def test_writer_failure_on_second_flush_compensates_all_files(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    """Writer-thread failure path: flush 1 commits, flush 2's index.add raises.
    The error must propagate and EVERY file row (including flush 1's) must be
    compensated, leaving the project indexable from scratch."""
    _write_words(tmp_path, ("a.py", "b.py", "c.py", "d.py"), 100)
    project = _project(tmp_path)
    runtime = _word_runtime(fake_runtime)
    inner_index = runtime.new_vector_index(runtime.embedder.ndim)
    runtime.new_vector_index = lambda ndim: _ExplodingOnNthAddIndex(inner_index, fail_on_call=2)  # type: ignore[method-assign,assignment,return-value]

    with pytest.raises(RuntimeError, match="simulated vector index write failure"):
        IndexEngine(runtime).index_project(project, AppConfig(model="fake", batch_size=32), IndexOptions())

    store = Store.open(project.metadata_db_path)
    try:
        assert store.get_meta("index_state") == "failed"
        assert store.get_files() == {}, "committed flush-1 files must be compensated too"
        assert store.counts(project.name).chunks_count == 0
    finally:
        store.close()

    # A rerun with a clean runtime must succeed and see all files as new again.
    stats = IndexEngine(_word_runtime(FakeRuntime())).index_project(project, AppConfig(model="fake", batch_size=32), IndexOptions())
    assert stats.files_indexed == 4
    assert stats.chunks_indexed == 400


def test_encode_failure_mid_pipeline_rolls_back_committed_flush(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    """Producer failure path: flush 1 is already handed to the writer when flush 2's
    encode raises. The writer's committed flush-1 rows must be compensated and the
    original encode error must propagate."""
    _write_words(tmp_path, ("a.py", "b.py", "c.py", "d.py"), 100)
    project = _project(tmp_path)
    runtime = _word_runtime(fake_runtime)

    calls = {"n": 0}
    real_encode = runtime.embedder.encode

    def _failing_encode(texts: list[str], *, is_query: bool = False, batch_size: int | None = None) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("encode exploded on second flush")
        return real_encode(texts, is_query=is_query, batch_size=batch_size)

    runtime.embedder.encode = _failing_encode  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="encode exploded on second flush"):
        IndexEngine(runtime).index_project(project, AppConfig(model="fake", batch_size=32), IndexOptions())

    store = Store.open(project.metadata_db_path)
    try:
        assert store.get_meta("index_state") == "failed"
        assert store.get_files() == {}
        assert store.counts(project.name).chunks_count == 0
    finally:
        store.close()


def test_pipelined_labels_are_deterministic_across_runs(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    """FIFO flush consumption must yield identical labels and file ids on every
    run — the pipelined path may reorder wall-clock work, never results."""
    _write_words(tmp_path, ("a.py", "b.py", "c.py", "d.py", "e.py"), 90)
    project = _project(tmp_path)
    runtime = _word_runtime(fake_runtime)

    def _sorted_labels() -> list[int]:
        store = Store.open(project.metadata_db_path, read_only=True)
        try:
            return sorted(int(row[0]) for row in store._conn.execute("SELECT label FROM chunks").fetchall())
        finally:
            store.close()

    IndexEngine(runtime).index_project(project, AppConfig(model="fake", batch_size=32), IndexOptions(rebuild=True))
    first = _sorted_labels()
    IndexEngine(runtime).index_project(project, AppConfig(model="fake", batch_size=32), IndexOptions(rebuild=True))
    second = _sorted_labels()

    assert first == second, "pipelined runs must produce identical label assignments"
    assert first == list(range(450)), "labels must be contiguous 0..N-1 in flush order"


def test_writer_overlaps_blocked_encode(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    """Overlap proof: while the second flush's encode is blocked, the writer must
    already have completed the first flush's index.add. A serial implementation
    times out here."""
    _write_words(tmp_path, ("a.py", "b.py", "c.py", "d.py"), 100)
    project = _project(tmp_path)
    runtime = _word_runtime(fake_runtime)
    gate = _OverlapGateEmbedder(runtime.embedder, block_on_call=2)
    runtime.embedder = gate  # type: ignore[assignment]
    inner_index = runtime.new_vector_index(runtime.embedder.ndim)
    runtime.new_vector_index = lambda ndim: _SpyIndex(inner_index, gate)  # type: ignore[method-assign,assignment,return-value]

    stats = IndexEngine(runtime).index_project(project, AppConfig(model="fake", batch_size=32), IndexOptions())

    assert stats.chunks_indexed == 400
    assert stats.vectors_added == 400
    store = Store.open(project.metadata_db_path)
    try:
        assert store.counts(project.name).chunks_count == 400
    finally:
        store.close()
