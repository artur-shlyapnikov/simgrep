from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from simgrep.clusters_engine import ClustersEngine
from simgrep.corpus import ChunkBatch, StoredChunk
from simgrep.errors import ClustersError
from simgrep.models import AppConfig, ClusterMember, ClustersOptions, ClustersOutcome, FileRole
from tests.conftest import FakeRuntime

_E0 = np.array([1, 0, 0, 0], dtype=np.float32)
_E1 = np.array([0, 1, 0, 0], dtype=np.float32)


_Spec = tuple[str, int | None, int | None, np.ndarray]


def _labels_of(members: tuple[ClusterMember, ...]) -> set[int]:
    return {member.label for member in members}


def _batch(base: Path, spec: list[_Spec]) -> ChunkBatch:
    """Aligned batch with one chunk per entry ``(rel_path, line_start, line_end, vector)``."""
    chunks = tuple(
        StoredChunk(
            label=idx,
            file_id=idx,
            file_path=base / fname,
            text="x",
            start_char=0,
            end_char=1,
            token_count=1,
            line_start=line_start,
            line_end=line_end,
            role=FileRole.source,
            language="python",
        )
        for idx, (fname, line_start, line_end, _vector) in enumerate(spec, start=1)
    )
    vectors = np.stack([vector.astype(np.float32) for *__, vector in spec]) if spec else np.zeros((0, 4), dtype=np.float32)
    return ChunkBatch(chunks=chunks, vectors=vectors, indexed_count=len(chunks))


def test_happy_path_finds_cross_file_cluster(tmp_path: Path) -> None:
    spec: list[_Spec] = [
        ("a.py", 1, 5, _E0),
        ("b.py", 10, 14, 2 * _E0),
        ("b.py", 40, 44, _E1),
    ]
    outcome = ClustersEngine(FakeRuntime()).run_batch(_batch(tmp_path, spec), ClustersOptions())

    assert outcome.chunks_scanned == 3
    assert outcome.total_found == 1
    assert len(outcome.clusters) == 1
    cluster = outcome.clusters[0]
    assert _labels_of(cluster.members) == {1, 2}
    assert cluster.score == pytest.approx(1.0)
    assert cluster.duplicated_lines == 10


def test_empty_index_yields_empty_outcome(tmp_path: Path) -> None:
    outcome = ClustersEngine(FakeRuntime()).run_batch(_batch(tmp_path, []), ClustersOptions())

    assert outcome == ClustersOutcome(clusters=(), total_found=0, chunks_scanned=0)


def test_label_missing_metadata_is_dropped(tmp_path: Path) -> None:
    # Labels 1/2 point along e1, labels 3/4 along e2. Label 4 has no line
    # metadata, so its would-be {3, 4} cluster must vanish.
    spec: list[_Spec] = [
        ("a.py", 1, 5, _E0),
        ("b.py", 10, 14, _E0),
        ("b.py", 40, 44, _E1),
        ("b.py", None, None, _E1),
    ]

    outcome = ClustersEngine(FakeRuntime()).run_batch(_batch(tmp_path, spec), ClustersOptions())

    assert outcome.chunks_scanned == 4
    assert len(outcome.clusters) == 1
    assert _labels_of(outcome.clusters[0].members) == {1, 2}


def test_max_chunks_guard_raises(tmp_path: Path) -> None:
    spec: list[_Spec] = [("a.py", 1, 5, _E0), ("b.py", 10, 14, _E0), ("b.py", 40, 44, _E1)]

    with pytest.raises(ClustersError):
        ClustersEngine(FakeRuntime()).run_batch(_batch(tmp_path, spec), replace(ClustersOptions(), max_chunks=2))


def test_top_cap_keeps_total_found(tmp_path: Path) -> None:
    spec: list[_Spec] = [
        ("a.py", 10, 14, np.array([1, 0, 0, 0], dtype=np.float32)),
        ("b.py", 20, 24, np.array([9, 0, 0, 0], dtype=np.float32)),
        ("c.py", 30, 34, np.array([0, 1, 0, 0], dtype=np.float32)),
        ("d.py", 40, 44, np.array([0, 7, 0, 0], dtype=np.float32)),
    ]

    outcome = ClustersEngine(FakeRuntime()).run_batch(_batch(tmp_path, spec), replace(ClustersOptions(), top=1))

    assert outcome.total_found == 2
    assert len(outcome.clusters) == 1
    assert _labels_of(outcome.clusters[0].members) == {1, 2}


def test_ephemeral_flow_via_clusters_path(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    duplicated = "alpha bravo charlie delta echo\n"
    (tmp_path / "a.py").write_text(duplicated, encoding="utf-8")
    (tmp_path / "b.py").write_text(duplicated, encoding="utf-8")
    # Tiny third chunk: its length-based fake vector must sit far below the threshold.
    (tmp_path / "c.py").write_text("x", encoding="utf-8")

    engine = ClustersEngine(fake_runtime)
    outcome = engine.clusters_path(tmp_path, AppConfig(model="fake"))

    assert outcome.chunks_scanned == 3
    assert outcome.total_found == 1
    paths = {Path(member.file_path).name for member in outcome.clusters[0].members}
    assert paths == {"a.py", "b.py"}


class _SpyReader:
    """CorpusReader stand-in that records close() calls."""

    def __init__(self, batch: ChunkBatch) -> None:
        self._batch = batch
        self.closed = False

    def snapshot(self) -> ChunkBatch:
        return self._batch

    def close(self) -> None:
        self.closed = True


def _patch_ephemeral(monkeypatch: pytest.MonkeyPatch, reader: _SpyReader) -> None:
    class _FakeCorpusAccess:
        def __init__(self, runtime: object) -> None:
            pass

        @contextmanager
        def open_ephemeral(
            self,
            paths: object,
            app_config: object,
            options: object | None = None,
        ) -> Iterator[_SpyReader]:
            try:
                yield reader
            finally:
                reader.close()

    monkeypatch.setattr("simgrep.clusters_engine.CorpusAccess", _FakeCorpusAccess)


def test_clusters_path_accepts_options_and_passes_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reader = _SpyReader(_batch(tmp_path, [("a.py", 1, 5, _E0[:2].copy()), ("b.py", 10, 14, _E0[:2].copy())]))
    _patch_ephemeral(monkeypatch, reader)

    # Options must reach run_batch: an invalid min_size trips validation there.
    with pytest.raises(ClustersError):
        ClustersEngine(FakeRuntime()).clusters_path(tmp_path, AppConfig(model="fake"), ClustersOptions(min_size=1))


def test_clusters_path_closes_corpus_on_ephemeral_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reader = _SpyReader(_batch(tmp_path, [("a.py", 1, 5, _E0[:2].copy()), ("b.py", 10, 14, _E0[:2].copy())]))
    _patch_ephemeral(monkeypatch, reader)

    outcome = ClustersEngine(FakeRuntime()).clusters_path(tmp_path, AppConfig(model="fake"))

    assert outcome.chunks_scanned == 2
    assert reader.closed is True
