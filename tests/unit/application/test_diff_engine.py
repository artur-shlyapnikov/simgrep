from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from simgrep.corpus import ChunkBatch, StoredChunk
from simgrep.diff_engine import DiffEngine
from simgrep.models import AppConfig, DiffEntry, DiffOptions, FileRole, FileRollup
from tests.conftest import FakeRuntime

E1 = np.array([[1, 0, 0, 0]], dtype=np.float32)


def _entry_of(outcome_added: tuple[DiffEntry, ...]) -> DiffEntry:
    assert len(outcome_added) == 1
    return outcome_added[0]


def _rollups_by_path(files: tuple[FileRollup, ...]) -> dict[str, FileRollup]:
    return {rollup.file_path: rollup for rollup in files}


def _batch(tree: Path, vectors: np.ndarray, *, missing_labels: frozenset[int] = frozenset()) -> ChunkBatch:
    """Aligned batch whose chunk k lives in <tree>/f<k>.py at line k*10; labels in
    ``missing_labels`` keep their ``indexed_count`` slot but lose their chunk row."""
    chunks: list[StoredChunk] = []
    kept_vectors: list[np.ndarray] = []
    for label, vector in enumerate(vectors, start=1):
        if label in missing_labels:
            continue
        chunks.append(
            StoredChunk(
                label=label,
                file_id=label,
                file_path=tree / f"f{label}.py",
                text=f"t{label}",
                start_char=0,
                end_char=3,
                token_count=1,
                line_start=label * 10,
                line_end=label * 10 + 4,
                role=FileRole.source,
                language="python",
            )
        )
        kept_vectors.append(vector.astype(np.float32))
    matrix = np.stack(kept_vectors) if kept_vectors else np.zeros((0, vectors.shape[1]), dtype=np.float32)
    return ChunkBatch(chunks=tuple(chunks), vectors=matrix.astype(np.float32, copy=False), indexed_count=vectors.shape[0])


def test_happy_path_two_trees_reports_matched_added_removed(tmp_path: Path) -> None:
    vecs_a = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    vecs_b = np.array([[3, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    batch_a = _batch(tmp_path / "a", vecs_a)
    batch_b = _batch(tmp_path / "b", vecs_b)

    outcome = DiffEngine(FakeRuntime()).run_batches(batch_a, batch_b, DiffOptions())

    assert outcome.matched == 1
    assert outcome.chunks_a == 3
    assert outcome.chunks_b == 2
    assert outcome.threshold == pytest.approx(0.8)
    added = _entry_of(outcome.added)
    assert added.label == 2
    assert added.file_path == str(tmp_path / "b" / "f2.py")
    assert (added.line_start, added.line_end) == (20, 24)
    assert [entry.label for entry in outcome.removed] == [2, 3]
    assert all(entry.file_path.startswith(str(tmp_path / "a")) for entry in outcome.removed)


def test_identical_trees_match_one_to_one(tmp_path: Path) -> None:
    vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    batch_a = _batch(tmp_path / "a", vecs)
    batch_b = _batch(tmp_path / "b", vecs.copy())

    outcome = DiffEngine(FakeRuntime()).run_batches(batch_a, batch_b, DiffOptions())

    assert outcome.matched == 2
    assert outcome.added == ()
    assert outcome.removed == ()


def test_empty_tree_a_reports_everything_in_b_as_added(tmp_path: Path) -> None:
    batch_a = _batch(tmp_path / "a", np.zeros((0, 4), dtype=np.float32))
    batch_b = _batch(tmp_path / "b", np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32))

    outcome = DiffEngine(FakeRuntime()).run_batches(batch_a, batch_b, DiffOptions())

    assert outcome.chunks_a == 0
    assert outcome.chunks_b == 2
    assert outcome.matched == 0
    assert outcome.removed == ()
    assert [entry.label for entry in outcome.added] == [1, 2]


def test_empty_tree_b_reports_everything_in_a_as_removed(tmp_path: Path) -> None:
    batch_a = _batch(tmp_path / "a", np.array([[1, 0, 0, 0]], dtype=np.float32))
    batch_b = _batch(tmp_path / "b", np.zeros((0, 4), dtype=np.float32))

    outcome = DiffEngine(FakeRuntime()).run_batches(batch_a, batch_b, DiffOptions())

    assert outcome.chunks_a == 1
    assert outcome.chunks_b == 0
    assert outcome.matched == 0
    assert outcome.added == ()
    assert [entry.label for entry in outcome.removed] == [1]


def test_label_missing_metadata_is_dropped_defensively(tmp_path: Path) -> None:
    # Label 2 exists in the index but has no chunk row in the store: it must
    # vanish entirely instead of surfacing as a bogus removal.
    vecs_a = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    batch_a = _batch(tmp_path / "a", vecs_a, missing_labels=frozenset({2}))
    batch_b = _batch(tmp_path / "b", E1.copy())

    outcome = DiffEngine(FakeRuntime()).run_batches(batch_a, batch_b, DiffOptions())

    assert outcome.matched == 1
    assert outcome.added == ()
    assert outcome.removed == ()
    assert outcome.chunks_a == 2


def test_top_cap_limits_lists_but_not_matched_or_files(tmp_path: Path) -> None:
    vecs_a = E1.copy()
    vecs_b = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    batch_a = _batch(tmp_path / "a", vecs_a)
    batch_b = _batch(tmp_path / "b", vecs_b)

    outcome = DiffEngine(FakeRuntime()).run_batches(batch_a, batch_b, replace(DiffOptions(), top=2))

    assert len(outcome.added) == 2
    assert outcome.matched == 1  # pre-cap
    assert outcome.chunks_b == 4
    # Rollups stay pre-cap: all five touched paths appear with their full counts.
    rolls = _rollups_by_path(outcome.files)
    assert len(rolls) == 5
    added_paths = [str(tmp_path / "b" / f"f{label}.py") for label in (2, 3, 4)]
    assert sum(rolls[p].added for p in added_paths) == 3
    assert rolls[str(tmp_path / "a" / "f1.py")].matched == 1


def test_rollup_attribution_sides_and_sort_order(tmp_path: Path) -> None:
    vecs_a = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    vecs_b = np.array([[5, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    batch_a = _batch(tmp_path / "a", vecs_a)
    batch_b = _batch(tmp_path / "b", vecs_b)

    outcome = DiffEngine(FakeRuntime()).run_batches(batch_a, batch_b, DiffOptions())

    rolls = _rollups_by_path(outcome.files)
    assert rolls[str(tmp_path / "a" / "f1.py")] == FileRollup(file_path=str(tmp_path / "a" / "f1.py"), added=0, removed=0, matched=1)
    assert rolls[str(tmp_path / "a" / "f2.py")] == FileRollup(file_path=str(tmp_path / "a" / "f2.py"), added=0, removed=1, matched=0)
    assert rolls[str(tmp_path / "b" / "f1.py")] == FileRollup(file_path=str(tmp_path / "b" / "f1.py"), added=0, removed=0, matched=1)
    assert rolls[str(tmp_path / "b" / "f2.py")] == FileRollup(file_path=str(tmp_path / "b" / "f2.py"), added=1, removed=0, matched=0)
    # Changed files (-(added+removed) == -1) come first, ties by file_path;
    # change-free matched files (key 0) come last.
    assert [rollup.file_path for rollup in outcome.files] == [
        str(tmp_path / "a" / "f2.py"),
        str(tmp_path / "b" / "f2.py"),
        str(tmp_path / "a" / "f1.py"),
        str(tmp_path / "b" / "f1.py"),
    ]


def test_default_options_are_applied_when_none_given(tmp_path: Path) -> None:
    batch_a = _batch(tmp_path / "a", E1.copy())
    batch_b = _batch(tmp_path / "b", E1.copy())

    outcome = DiffEngine(FakeRuntime()).run_batches(batch_a, batch_b, None)

    assert outcome.threshold == pytest.approx(DiffOptions().threshold)


def test_threshold_flows_through_to_matching(tmp_path: Path) -> None:
    near = np.array([[0.9, float(np.sqrt(0.19)), 0, 0]], dtype=np.float32)  # cosine 0.9 vs E1
    batch_a = _batch(tmp_path / "a", E1.copy())
    batch_b = _batch(tmp_path / "b", near)

    matched_outcome = DiffEngine(FakeRuntime()).run_batches(batch_a, batch_b, DiffOptions(threshold=0.8))
    strict_outcome = DiffEngine(FakeRuntime()).run_batches(batch_a, batch_b, replace(DiffOptions(), threshold=0.95))

    assert matched_outcome.matched == 1
    assert strict_outcome.matched == 0
    assert strict_outcome.threshold == pytest.approx(0.95)
    assert strict_outcome.removed[0].file_path == str(tmp_path / "a" / "f1.py")
    assert strict_outcome.added[0].file_path == str(tmp_path / "b" / "f1.py")


class _SpyReader:
    """CorpusReader stand-in that records close() calls."""

    def __init__(self, batch: ChunkBatch) -> None:
        self._batch = batch
        self.closed = False

    def snapshot(self) -> ChunkBatch:
        return self._batch

    def close(self) -> None:
        self.closed = True


def _patch_ephemeral(monkeypatch: pytest.MonkeyPatch, readers: dict[str, _SpyReader]) -> None:
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


def test_diff_paths_closes_both_handles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "a").mkdir(exist_ok=True)
    (tmp_path / "b").mkdir(exist_ok=True)
    readers = {
        "a": _SpyReader(_batch(tmp_path / "a", E1.copy())),
        "b": _SpyReader(_batch(tmp_path / "b", E1.copy())),
    }
    _patch_ephemeral(monkeypatch, readers)

    outcome = DiffEngine(FakeRuntime()).diff_paths(tmp_path / "a", tmp_path / "b", AppConfig(model="fake"), DiffOptions())

    assert outcome.matched == 1
    assert readers["a"].closed is True
    assert readers["b"].closed is True


def test_diff_paths_closes_first_handle_when_second_build_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "a").mkdir(exist_ok=True)
    (tmp_path / "b").mkdir(exist_ok=True)
    spy_a = _SpyReader(_batch(tmp_path / "a", E1.copy()))

    class _ExplodingCorpusAccess:
        def __init__(self, runtime: object) -> None:
            pass

        @contextmanager
        def open_ephemeral(
            self,
            paths: Sequence[Path],
            app_config: AppConfig,
            options: object | None = None,
        ) -> Iterator[_SpyReader]:
            if Path(list(paths)[0]).name != "a":
                raise RuntimeError("boom")
            try:
                yield spy_a
            finally:
                spy_a.close()

    monkeypatch.setattr("simgrep.diff_engine.CorpusAccess", _ExplodingCorpusAccess)

    with pytest.raises(RuntimeError, match="boom"):
        DiffEngine(FakeRuntime()).diff_paths(tmp_path / "a", tmp_path / "b", AppConfig(model="fake"))

    assert spy_a.closed is True


def test_ephemeral_flow_renamed_file_is_invisible(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    body = "alpha bravo charlie delta echo\n"
    tree_a = tmp_path / "a"
    tree_b = tmp_path / "b"
    tree_a.mkdir()
    tree_b.mkdir()
    (tree_a / "old_name.py").write_text(body, encoding="utf-8")
    (tree_b / "new_name.py").write_text(body, encoding="utf-8")
    # Different length -> a far-apart fake vector -> genuinely added chunk.
    (tree_b / "brand_new.py").write_text("zz", encoding="utf-8")

    outcome = DiffEngine(fake_runtime).diff_paths(tree_a, tree_b, AppConfig(model="fake"))

    assert outcome.matched == 1
    assert outcome.chunks_a == 1
    assert outcome.chunks_b == 2
    assert [Path(entry.file_path).name for entry in outcome.added] == ["brand_new.py"]
    assert outcome.removed == ()
