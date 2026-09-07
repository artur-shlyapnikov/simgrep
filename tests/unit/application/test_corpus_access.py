"""Contract tests for the deep corpus boundary (simgrep.corpus).

Pins the invariants callers previously had to reconstruct by hand: label/vector/metadata
alignment, rank-preserving typed lookup, ephemeral cleanup, and — critically — that a
persistent reader session excludes concurrent indexing writers for its whole lifetime.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from simgrep.corpus import ChunkBatch, CorpusAccess, CorpusReader, StoredChunk, ephemeral_options
from simgrep.errors import SearchError
from simgrep.indexing import IndexEngine
from simgrep.models import (
    SCHEMA_VERSION,
    AppConfig,
    ChunkRecord,
    FileRecord,
    FileRole,
    FreshnessMode,
    IndexOptions,
    ProjectConfig,
    TermRecord,
)
from simgrep.project import init_project
from simgrep.store import Store
from tests.conftest import FakeRuntime, FakeVectorIndex

_E0 = np.array([1, 0, 0, 0], dtype=np.float32)
_E1 = np.array([0, 1, 0, 0], dtype=np.float32)


def _reader_with_rows(tmp_path: Path, *, drop_label: int | None = None) -> CorpusReader:
    """Reader over an in-memory store + fake index with two aligned chunks.

    ``drop_label`` omits that label's metadata row to simulate a metadata/vector mismatch.
    """
    store = Store.memory()
    file_id = store.upsert_file(FileRecord(id=0, path=tmp_path / "a.py", size_bytes=16, mtime_ns=1, role=FileRole.source, language="python"))
    records = [
        ChunkRecord(label=1, file_id=file_id, text="alpha", start_char=0, end_char=5, token_count=2, line_start=1, line_end=2),
        ChunkRecord(label=2, file_id=file_id, text="beta", start_char=6, end_char=10, token_count=2, line_start=3, line_end=4),
    ]
    store.insert_terms(
        [
            TermRecord(label=1, term="alpha", field="chunk", tf=2, weight=1.0),
            TermRecord(label=2, term="beta", field="chunk", tf=2, weight=1.0),
        ]
    )
    store.insert_chunks([record for record in records if drop_label is None or record.label != drop_label])
    index = FakeVectorIndex(ndim=4)
    index.add(keys=np.array([1, 2], dtype=np.int64), vecs=np.stack([_E0, _E1]))
    return CorpusReader(store=store, index=index, base_path=tmp_path)

    # ------------------------------------------------------------------ alignment

    batch = _reader_with_rows(tmp_path).snapshot()
    assert isinstance(batch, ChunkBatch)
    assert batch.indexed_count == 2
    assert [chunk.label for chunk in batch.chunks] == [1, 2]
    assert np.array_equal(batch.vectors[0], _E0)
    assert np.array_equal(batch.vectors[1], _E1)
    assert len(batch.vectors) == len(batch.chunks)


def test_snapshot_drops_labels_without_metadata_but_reports_indexed_count(tmp_path: Path) -> None:
    batch = _reader_with_rows(tmp_path, drop_label=2).snapshot()
    assert batch.indexed_count == 2  # index still held both labels
    assert [chunk.label for chunk in batch.chunks] == [1]
    assert np.array_equal(batch.vectors[0], _E0)  # row stays aligned with the surviving chunk


def test_snapshot_on_empty_index_is_empty_not_error(tmp_path: Path) -> None:
    batch = CorpusReader(store=Store.memory(), index=FakeVectorIndex(ndim=4), base_path=tmp_path).snapshot()
    assert len(batch) == 0
    assert batch.vectors.shape == (0, 4)


def test_lookup_preserves_caller_rank_and_returns_typed_chunks(tmp_path: Path) -> None:
    reader = _reader_with_rows(tmp_path)
    chunks = reader.lookup([2, 1])
    assert [chunk.label for chunk in chunks] == [2, 1]
    first = chunks[0]
    assert isinstance(first, StoredChunk)
    assert first.text == "beta"
    assert first.file_path == tmp_path / "a.py"
    assert first.role.value == "source"
    assert (first.line_start, first.line_end) == (3, 4)


def test_lexical_pairs_typed_chunks_with_scores(tmp_path: Path) -> None:
    reader = _reader_with_rows(tmp_path)
    hits = reader.lexical(["alpha"], limit=5)
    assert [chunk.label for chunk, _score in hits] == [1]
    assert hits[0][1] > 0.0


def test_ephemeral_options_default_to_app_patterns() -> None:
    config = AppConfig(file_patterns=("*.md",))
    opts = ephemeral_options(config)
    assert opts.scan.patterns == ("*.md",)
    assert opts.scan.include_globs == ()
    assert opts.scan.max_file_size_bytes == config.max_file_size_bytes


# ------------------------------------------------------------------ lifecycle


def _project_with_one_file(root: Path, runtime: FakeRuntime) -> tuple[ProjectConfig, AppConfig]:
    root.mkdir(parents=True)
    config = AppConfig(model="fake")
    project = init_project(root, config, name="corpus-contract", yes=True)
    (root / "mod.py").write_text("x" * 40, encoding="utf-8")
    IndexEngine(runtime).index_project(project, config, IndexOptions(rebuild=True))
    return project, config


def test_open_ephemeral_yields_reader_and_closes_afterwards(tmp_path: Path) -> None:
    target = tmp_path / "src"
    target.mkdir()
    (target / "m.py").write_text("y" * 30, encoding="utf-8")
    with CorpusAccess(FakeRuntime()).open_ephemeral([target], AppConfig(model="fake")) as reader:
        assert reader.chunk_count == 1
        inner = reader
    with pytest.raises(Exception):  # noqa: B017 - duckdb raises on closed connection
        inner.lookup([1])


def test_open_ephemeral_cleans_up_when_body_raises(tmp_path: Path) -> None:
    target = tmp_path / "src"
    target.mkdir()
    (target / "m.py").write_text("y" * 30, encoding="utf-8")

    def _boom() -> None:
        with CorpusAccess(FakeRuntime()).open_ephemeral([target], AppConfig(model="fake")):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _boom()


_LOCK_PROBE = (
    "import sys\n"
    "from filelock import FileLock\n"
    "try:\n"
    "    FileLock(sys.argv[1], timeout=2).acquire()\n"
    "except Exception:\n"
    "    raise SystemExit(1)\n"
    "raise SystemExit(0)\n"
)


def _probe_lock(lock_path: Path) -> int:
    return subprocess.run(  # noqa: S603 - fixed argv, test-only probe
        [sys.executable, "-c", _LOCK_PROBE, str(lock_path)], capture_output=True, timeout=60
    ).returncode


def test_indexing_writer_excluded_for_whole_reader_session(tmp_path: Path) -> None:
    """Regression: no indexing writer may mutate artifacts under an active reader."""
    runtime = FakeRuntime()
    project, config = _project_with_one_file(tmp_path / "proj", runtime)
    access = CorpusAccess(runtime)
    with access.open_project(project, config, freshness=FreshnessMode.skip) as reader:
        assert reader.chunk_count == 1
        assert _probe_lock(project.index_lock_path) == 1  # writer cannot take the lock mid-session
    assert _probe_lock(project.index_lock_path) == 0  # released once the session ends


def test_open_project_missing_artifacts_raise_search_error(tmp_path: Path) -> None:
    project = ProjectConfig(
        schema_version=SCHEMA_VERSION,
        name="ghost",
        root=tmp_path,
        indexed_paths=(tmp_path,),
        model="fake",
        chunk_size=128,
        chunk_overlap=16,
    )
    (tmp_path / ".simgrep").mkdir()

    def _open() -> None:
        with CorpusAccess(FakeRuntime()).open_project(project, AppConfig(model="fake")):
            pass  # pragma: no cover - body never reached

    with pytest.raises(SearchError):
        _open()


def test_open_project_freshness_check_flags_stale_index(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    project, config = _project_with_one_file(tmp_path / "proj", runtime)
    (project.root / "mod.py").write_text("totally different " * 4, encoding="utf-8")  # stat-stale

    def _open() -> None:
        with CorpusAccess(runtime).open_project(project, config, freshness=FreshnessMode.check):
            pass  # pragma: no cover

    with pytest.raises(SearchError, match="stale"):
        _open()
