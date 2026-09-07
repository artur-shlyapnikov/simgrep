from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from simgrep.corpus import ChunkBatch, CorpusAccess
from simgrep.diffing import match_trees
from simgrep.errors import DiffError
from simgrep.models import (
    AppConfig,
    DiffEntry,
    DiffOptions,
    DiffOutcome,
    FileRollup,
)


class DiffEngine:
    """Semantic tree diff over two ephemeral chunk corpora."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def diff_paths(
        self,
        path_a: Path,
        path_b: Path,
        app_config: AppConfig,
        options: DiffOptions | None = None,
    ) -> DiffOutcome:
        opts = options if options is not None else DiffOptions()
        if not 0 < opts.threshold <= 1.0:
            raise DiffError(
                f"Threshold must satisfy 0 < threshold <= 1, got {opts.threshold}.",
                hint="Pass a similarity threshold from the open interval (0.0, 1.0], e.g. 0.85.",
            )
        for path in (path_a, path_b):
            if not path.exists():
                raise DiffError(f"Path not found: {path}", hint="Check the path and try again.")
        access = CorpusAccess(self.runtime)
        with access.open_ephemeral([path_a], app_config) as corpus_a:
            with access.open_ephemeral([path_b], app_config) as corpus_b:
                return self.run_batches(corpus_a.snapshot(), corpus_b.snapshot(), options)

    def run_batches(self, batch_a: ChunkBatch, batch_b: ChunkBatch, options: DiffOptions | None = None) -> DiffOutcome:
        opts = options if options is not None else DiffOptions()
        entries_a, vectors_a, kept_a, total_a = _collect(batch_a)
        entries_b, vectors_b, kept_b, total_b = _collect(batch_b)
        pairs, unmatched_a, unmatched_b = match_trees(vectors_a, vectors_b, kept_a, kept_b, opts)

        removed = sorted(
            (entries_a[label] for label in unmatched_a),
            key=lambda entry: (entry.file_path, entry.line_start, entry.label),
        )
        added = sorted(
            (entries_b[label] for label in unmatched_b),
            key=lambda entry: (entry.file_path, entry.line_start, entry.label),
        )
        return DiffOutcome(
            added=tuple(added[: opts.top]),
            removed=tuple(removed[: opts.top]),
            matched=len(pairs),
            files=_rollups(entries_a, entries_b, pairs, added, removed),
            chunks_a=total_a,
            chunks_b=total_b,
            threshold=opts.threshold,
        )


def _collect(batch: ChunkBatch) -> tuple[dict[int, DiffEntry], np.ndarray, list[int], int]:
    """Gather metadata entries; labels without store rows are dropped defensively."""
    if len(batch) == 0:
        return {}, np.zeros((0, batch.vectors.shape[1]), dtype=np.float32), [], batch.indexed_count
    entries: dict[int, DiffEntry] = {}
    for chunk in batch.chunks:
        entries[chunk.label] = DiffEntry(
            label=chunk.label,
            file_path=str(chunk.file_path),
            # Historical dict flow passed raw store values; None stays None here.
            line_start=chunk.line_start,  # type: ignore[arg-type]
            line_end=chunk.line_end,  # type: ignore[arg-type]
        )
    return entries, batch.vectors, [chunk.label for chunk in batch.chunks], batch.indexed_count


def _rollups(
    entries_a: dict[int, DiffEntry],
    entries_b: dict[int, DiffEntry],
    pairs: list[tuple[int, int, float]],
    added: list[DiffEntry],
    removed: list[DiffEntry],
) -> tuple[FileRollup, ...]:
    counts: dict[str, list[int]] = {}

    def bump(path: str, slot: int) -> None:
        slot_counts = counts.setdefault(path, [0, 0, 0])
        slot_counts[slot] += 1

    for entry in added:
        bump(entry.file_path, 0)
    for entry in removed:
        bump(entry.file_path, 1)
    for label_a, label_b, _score in pairs:
        bump(entries_a[label_a].file_path, 2)
        bump(entries_b[label_b].file_path, 2)
    rollups = [FileRollup(file_path=path, added=slot[0], removed=slot[1], matched=slot[2]) for path, slot in counts.items()]
    return tuple(sorted(rollups, key=lambda rollup: (-(rollup.added + rollup.removed), rollup.file_path)))
