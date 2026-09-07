"""Round-16 edge coverage for DiffEngine (simgrep/diff_engine.py).

- diff_paths closes BOTH corpora when the matching phase fails (not only on
  success or second-build failure);
- added/removed use the full (file_path, line_start, label) sort key and the
  top cap keeps the SORTED prefix, never the first-built entries;
- rollups stay pre-cap while the entry tuples are capped;
- dropping a middle label keeps the remaining vector->label gather aligned.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from simgrep.corpus import ChunkBatch, StoredChunk
from simgrep.diff_engine import DiffEngine
from simgrep.errors import DiffError
from simgrep.models import AppConfig, DiffOptions, FileRole
from tests.conftest import FakeRuntime

E1 = np.array([[1, 0, 0, 0]], dtype=np.float32)


def _batch(
    tmp_path: Path,
    name: str,
    vectors: np.ndarray,
    *,
    missing_labels: frozenset[int] = frozenset(),
    file_of: Callable[[int], str] | None = None,
    line_of: Callable[[int], int] | None = None,
) -> ChunkBatch:
    """Aligned batch whose chunk k lives in <tree>/<file_of(k)> at line <line_of(k)>."""
    tree = tmp_path / name
    chunks: list[StoredChunk] = []
    kept_vectors: list[np.ndarray] = []
    for label, vector in enumerate(vectors, start=1):
        if label in missing_labels:
            continue
        file_name = file_of(label) if file_of is not None else f"f{label}.py"
        line_start = line_of(label) if line_of is not None else label * 10
        chunks.append(
            StoredChunk(
                label=label,
                file_id=label,
                file_path=tree / file_name,
                text=f"t{label}",
                start_char=0,
                end_char=3,
                token_count=1,
                line_start=line_start,
                line_end=line_start + 4,
                role=FileRole.source,
                language="python",
            )
        )
        kept_vectors.append(vector.astype(np.float32))
    matrix = np.stack(kept_vectors) if kept_vectors else np.zeros((0, vectors.shape[1]), dtype=np.float32)
    return ChunkBatch(chunks=tuple(chunks), vectors=matrix.astype(np.float32, copy=False), indexed_count=vectors.shape[0])


class _SpyReader:
    """CorpusReader stand-in that records close() calls."""

    def __init__(self, batch: ChunkBatch) -> None:
        self._batch = batch
        self.closed = False

    def snapshot(self) -> ChunkBatch:
        return self._batch

    def close(self) -> None:
        self.closed = True


def test_diff_paths_closes_both_handles_when_matching_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "a").mkdir(exist_ok=True)
    (tmp_path / "b").mkdir(exist_ok=True)
    readers = {
        "a": _SpyReader(_batch(tmp_path, "a", E1.copy())),
        "b": _SpyReader(_batch(tmp_path, "b", E1.copy())),
    }

    class _FakeCorpusAccess:
        def __init__(self, runtime: object) -> None:
            pass

        @contextmanager
        def open_ephemeral(
            self,
            paths: Sequence[Path],
            app_config: AppConfig,
            options: object | None = None,
        ) -> Iterator[_SpyReader]:
            reader = readers[Path(list(paths)[0]).name]
            try:
                yield reader
            finally:
                reader.close()

    monkeypatch.setattr("simgrep.diff_engine.CorpusAccess", _FakeCorpusAccess)

    with pytest.raises(DiffError):
        # max_chunks=1 makes the MATCHING phase raise after both builds succeed.
        DiffEngine(FakeRuntime()).diff_paths(
            tmp_path / "a",
            tmp_path / "b",
            AppConfig(),
            DiffOptions(max_chunks=1),
        )

    assert readers["a"].closed is True
    assert readers["b"].closed is True


def test_added_sorted_by_line_within_one_file_and_cap_applied_after_sort(tmp_path: Path) -> None:
    batch_a = _batch(tmp_path, "a", E1.copy())
    batch_b = _batch(
        tmp_path,
        "b",
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32),
        file_of=lambda label: "same.py",
        line_of=lambda label: {1: 90, 2: 10, 3: 50}[label],
    )

    outcome = DiffEngine(FakeRuntime()).run_batches(batch_a, batch_b, replace(DiffOptions(), top=2))

    assert [(entry.label, entry.line_start) for entry in outcome.added] == [(2, 10), (3, 50)]


def test_top_cap_keeps_sorted_prefix_for_removed_too(tmp_path: Path) -> None:
    batch_a = _batch(
        tmp_path,
        "a",
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32),
        file_of=lambda label: "same.py",
        line_of=lambda label: {1: 90, 2: 10, 3: 50}[label],
    )
    batch_b = _batch(tmp_path, "b", np.array([[0, 0, 0, 1]], dtype=np.float32), file_of=lambda label: "same.py")

    outcome = DiffEngine(FakeRuntime()).run_batches(batch_a, batch_b, replace(DiffOptions(), top=1))

    assert outcome.matched == 0
    assert len(outcome.removed) == 1
    assert outcome.removed[0].file_path == str(tmp_path / "a" / "same.py")
    assert outcome.removed[0].line_start == 10
    rollups = {rollup.file_path: rollup for rollup in outcome.files}
    assert rollups[str(tmp_path / "a" / "same.py")].removed == 3


def test_dropped_middle_label_keeps_remaining_vectors_aligned(tmp_path: Path) -> None:
    batch_a = _batch(
        tmp_path,
        "a",
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32),
        missing_labels=frozenset({2}),
    )
    batch_b = _batch(tmp_path, "b", np.array([[0, 0, 1, 0], [1, 0, 0, 0]], dtype=np.float32))

    outcome = DiffEngine(FakeRuntime()).run_batches(batch_a, batch_b)

    assert outcome.matched == 2
    assert outcome.added == ()
    assert outcome.removed == ()
