from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from simgrep.indexing import IndexEngine
from simgrep.models import SCHEMA_VERSION, AppConfig, Chunk, IndexOptions, IndexState, ProjectConfig
from simgrep.store import Store
from tests.conftest import FakeRuntime, FakeVectorIndex


def _project(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path,), "fake", 128, 20)


class _ExplodingIndex:
    """Delegates everything to a real index but makes index.add fail."""

    def __init__(self, inner: FakeVectorIndex) -> None:
        self._inner = inner

    def __len__(self) -> int:
        return len(self._inner)

    def add(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("simulated vector index write failure")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class ExplodingIndexRuntime(FakeRuntime):
    def new_vector_index(self, ndim: int) -> Any:
        return _ExplodingIndex(super().new_vector_index(ndim))


class _LinePerChunkChunker:
    """One chunk per non-empty line, forcing pending volume over the flush threshold."""

    def chunk(self, text: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        offset = 0
        for line in text.splitlines(keepends=True):
            if line.strip():
                chunks.append(Chunk(id=-1, file_id=-1, text=line, start=offset, end=offset + len(line), tokens=max(1, len(line.split()))))
            offset += len(line)
        return chunks


class _FailOnNthAddIndex:
    """Real index until the Nth add call, then explodes."""

    def __init__(self, inner: FakeVectorIndex, fail_on_call: int) -> None:
        self._inner = inner
        self._fail_on_call = fail_on_call
        self.add_calls = 0

    def __len__(self) -> int:
        return len(self._inner)

    def add(self, *args: Any, **kwargs: Any) -> None:
        self.add_calls += 1
        if self.add_calls >= self._fail_on_call:
            raise RuntimeError("simulated mid-run vector index write failure")
        self._inner.add(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _MidRunFailureRuntime(FakeRuntime):
    def __init__(self, fail_on_call: int) -> None:
        super().__init__()
        self.chunker: Any = _LinePerChunkChunker()
        self._fail_on_call = fail_on_call

    def new_vector_index(self, ndim: int) -> Any:
        return _FailOnNthAddIndex(super().new_vector_index(ndim), self._fail_on_call)


class TestFailedRunCompensation:
    def test_index_add_failure_rolls_back_store_rows_and_next_run_recovers(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
        (tmp_path / "b.py").write_text("gamma delta", encoding="utf-8")
        project = _project(tmp_path)

        # With the stock single-chunk-per-file chunker all files share ONE flush, so this
        # test covers the first-flush-failure partition; the cross-flush partition is owned
        # by TestMidRunFlushCompensation below.
        engine = IndexEngine(ExplodingIndexRuntime())
        with pytest.raises(RuntimeError, match="simulated vector index write failure"):
            engine.index_project(project, AppConfig(model="fake"), IndexOptions())

        store = Store.open(project.metadata_db_path)
        try:
            assert store.get_files() == {}, f"Expected no file rows after compensation, got {store.get_files()}"
            files_row = store._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            assert files_row is not None
            assert files_row[0] == 0, "Expected no chunk rows after compensation"
            assert store.get_meta("index_state") == IndexState.failed.value
        finally:
            store.close()

        stats = IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions())
        assert stats.files_skipped_unchanged == 0, "Compensated files must be planned as new, not skipped"
        assert stats.files_indexed == 2
        assert stats.chunks_indexed == 2

        reopened = Store.open(project.metadata_db_path)
        try:
            chunk_labels = {row[0] for row in reopened._conn.execute("SELECT label FROM chunks").fetchall()}
        finally:
            reopened.close()
        index = fake_runtime.new_vector_index(4)
        index.load(project.vector_index_path)
        assert chunk_labels == set(index.data.keys()), f"Chunks {chunk_labels} vs vectors {set(index.data.keys())}"


class TestDivergenceSelfHeal:
    def test_store_chunks_without_on_disk_vectors_trigger_full_rebuild(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        (tmp_path / "a.py").write_text("alpha beta gamma", encoding="utf-8")
        (tmp_path / "b.py").write_text("delta epsilon", encoding="utf-8")
        project = _project(tmp_path)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

        empty = fake_runtime.new_vector_index(4)
        empty.save(project.vector_index_path)

        stats = IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions())

        assert stats.files_skipped_unchanged == 0, "Diverged store must force a full rebuild, not skip unchanged files"
        assert stats.files_indexed == 2
        assert stats.index_mutated is True

        reopened = Store.open(project.metadata_db_path)
        try:
            chunk_labels = {row[0] for row in reopened._conn.execute("SELECT label FROM chunks").fetchall()}
        finally:
            reopened.close()
        healed = fake_runtime.new_vector_index(4)
        healed.load(project.vector_index_path)
        assert chunk_labels == set(healed.data.keys()), f"Chunks {chunk_labels} vs vectors {set(healed.data.keys())}"

    def test_equal_chunk_and_vector_counts_reuse_index_without_rebuild(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        (tmp_path / "a.py").write_text("alpha beta gamma", encoding="utf-8")
        (tmp_path / "b.py").write_text("delta epsilon", encoding="utf-8")
        project = _project(tmp_path)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

        # Ground the boundary precondition: exactly equal counts, not accidentally diverged.
        store = Store.open(project.metadata_db_path)
        probe = fake_runtime.new_vector_index(4)
        try:
            assert store.counts(project.name).chunks_count > 0
            probe.load(project.vector_index_path)
            assert store.counts(project.name).chunks_count == len(probe)
        finally:
            store.close()

        stats = IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions())

        assert stats.files_skipped_unchanged == 2, "Equal counts is the healthy state; must reuse, not rebuild"
        assert stats.files_indexed == 0
        assert stats.index_mutated is False, "Healthy reuse must not rewrite artifacts"

    def test_dry_run_on_diverged_store_does_not_clear_or_rebuild(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        (tmp_path / "a.py").write_text("alpha beta gamma", encoding="utf-8")
        (tmp_path / "b.py").write_text("delta epsilon", encoding="utf-8")
        project = _project(tmp_path)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))
        empty = fake_runtime.new_vector_index(4)
        empty.save(project.vector_index_path)

        stats = IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(dry_run=True))

        assert stats.files_seen == 2
        assert stats.files_indexed == 0
        reopened = Store.open(project.metadata_db_path)
        try:
            assert reopened.counts(project.name).chunks_count == 2, "Dry-run must leave the diverged store untouched"
            assert reopened.get_meta("index_state") == IndexState.ready.value
        finally:
            reopened.close()

    def test_missing_vector_artifact_with_nonempty_store_triggers_full_rebuild(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
        project = _project(tmp_path)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))
        project.vector_index_path.unlink()

        stats = IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions())

        assert stats.files_skipped_unchanged == 0
        assert stats.files_indexed == 1
        healed = fake_runtime.new_vector_index(4)
        healed.load(project.vector_index_path)
        assert len(healed) == 1


class TestMidRunFlushCompensation:
    def test_failure_in_later_flush_also_compensates_files_from_earlier_flushes(self, tmp_path: Path) -> None:
        (tmp_path / "big.py").write_text("\n".join(f"line {i} alpha" for i in range(300)) + "\n", encoding="utf-8")
        (tmp_path / "small.py").write_text("gamma delta\n", encoding="utf-8")
        project = _project(tmp_path)

        engine = IndexEngine(_MidRunFailureRuntime(fail_on_call=2))
        with pytest.raises(RuntimeError, match="mid-run vector index write failure"):
            engine.index_project(project, AppConfig(model="fake", batch_size=32), IndexOptions())

        store = Store.open(project.metadata_db_path)
        try:
            assert store.get_files() == {}, "Files committed by earlier flushes must also be compensated"
            counts = store.counts(project.name)
            assert counts.chunks_count == 0
            assert store.get_meta("index_state") == IndexState.failed.value
        finally:
            store.close()


class TestCompensationNeverMasks:
    def test_failing_delete_during_compensation_preserves_original_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
        project = _project(tmp_path)

        def exploding_delete(self: Store, file_id: int) -> list[int]:
            raise ValueError("simulated compensation boom")

        monkeypatch.setattr(Store, "delete_file", exploding_delete)
        engine = IndexEngine(ExplodingIndexRuntime())
        with pytest.raises(RuntimeError, match="simulated vector index write failure"):
            engine.index_project(project, AppConfig(model="fake"), IndexOptions())

        store = Store.open(project.metadata_db_path)
        try:
            assert store.get_meta("index_state") == IndexState.failed.value, "failed marker must be set even when compensation itself raised"
        finally:
            store.close()
