"""Engine for ``simgrep debt``: the semantic debt-marker radar.
Batch algorithm over :mod:`simgrep.debt` fed a prepared corpus, attaching git
ages. Corpus acquisition (scope, runtime, freshness) lives in the application
layer (:func:`simgrep.execution.open_resolved_corpus`), never here.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from simgrep.corpus import ChunkBatch
from simgrep.debt import build_report, scan_text
from simgrep.errors import DebtError
from simgrep.models import (
    AppConfig,
    DebtOptions,
    DebtReport,
    FreshnessMode,
    ProjectConfig,
)

_SCOPE_HINT = "Narrow the scope (e.g. a subdirectory)."
_NO_AGES_HINT = "Run inside a git repository so --max-age can compare last-commit dates."


class DebtEngine:
    """Cluster debt markers into themes and gate CI on their ages (batch-only)."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def debt_project(
        self,
        project: ProjectConfig,
        app_config: AppConfig,
        options: DebtOptions | None = None,
        freshness: FreshnessMode = FreshnessMode.auto,
    ) -> DebtReport:
        """Compatibility shim; transports go through simgrep.execution.execute_debt."""
        from simgrep.corpus import CorpusAccess

        opts = options if options is not None else DebtOptions()
        with CorpusAccess(self.runtime).open_project(project, app_config, freshness=freshness) as corpus:
            return self.run_batch(corpus.snapshot(), project.root, opts)

    def debt_path(self, path: Path, app_config: AppConfig, options: DebtOptions | None = None) -> DebtReport:
        """Compatibility shim; transports go through simgrep.execution.execute_debt."""
        from simgrep.corpus import CorpusAccess

        opts = options if options is not None else DebtOptions()
        with CorpusAccess(self.runtime).open_ephemeral([path], app_config) as corpus:
            return self.run_batch(corpus.snapshot(), path, opts)

    def run_batch(self, batch: ChunkBatch, root: Path, options: DebtOptions, *, now_epoch: float | None = None) -> DebtReport:
        # `now_epoch` feeds rich age display downstream only; reports store raw epochs.
        _now = time.time() if now_epoch is None else now_epoch

        raw_paths = [str(chunk.file_path) for chunk in batch.chunks]
        chunk_texts = [chunk.text for chunk in batch.chunks]
        line_starts = [chunk.line_start for chunk in batch.chunks]
        vectors = np.asarray(batch.vectors, dtype=np.float32)
        chunks_scanned = len(batch)
        if chunks_scanned > options.max_chunks:
            raise DebtError(
                f"Corpus too large for a debt scan: {chunks_scanned} chunks exceed max_chunks={options.max_chunks}.",
                hint=_SCOPE_HINT,
            )

        display_of_raw = {raw: self._display(raw, root) for raw in raw_paths}
        candidates: list[tuple[int, str, int, str, str]] = []
        kept_vectors: list[np.ndarray] = []
        texts_by_row: dict[int, str] = {}
        kept_pos = 0
        for raw_path, text, line_start, vector in zip(raw_paths, chunk_texts, line_starts, vectors, strict=True):
            hits = scan_text(text, 1 if line_start is None else int(line_start))
            if not hits:
                continue
            display = display_of_raw[raw_path]
            for line, marker, snippet in hits:
                candidates.append((kept_pos, display, line, marker, snippet))
            kept_vectors.append(vector)
            texts_by_row[kept_pos] = text
            kept_pos += 1

        if not candidates:
            return DebtReport(
                themes=(),
                scattered=0,
                markers_found=0,
                chunks_scanned=chunks_scanned,
                truncated=False,
                threshold=options.threshold,
                max_age_days=options.max_age_days,
                passed=None if options.max_age_days is None else True,
            )

        epochs_by_path: dict[str, int | None] = {}
        epoch_cache: dict[str, int | None] = {}
        raw_of_display = {display: raw for raw, display in display_of_raw.items()}
        for _, display, _, _, _ in candidates:
            if display in epochs_by_path:
                continue
            raw_path = raw_of_display[display]
            if raw_path not in epoch_cache:
                epoch_cache[raw_path] = self._git_epoch(root, raw_path)
            epochs_by_path[display] = epoch_cache[raw_path]

        report = build_report(candidates, np.stack(kept_vectors).astype(np.float32, copy=False), texts_by_row, epochs_by_path, options)
        if options.max_age_days is not None and all(epoch is None for epoch in epochs_by_path.values()):
            raise DebtError("no git ages available for any file carrying debt markers.", hint=_NO_AGES_HINT)
        # build_report only sees candidate rows; restore the true scanned-chunk count.
        return replace(report, chunks_scanned=chunks_scanned)

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _git_epoch(root: Path, raw_path: str) -> int | None:
        """Last-commit epoch for ``raw_path``; None outside git or for untracked files."""
        proc = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%ct", "--", raw_path], capture_output=True, text=True)
        stdout = proc.stdout.strip()
        if proc.returncode != 0 or not stdout.isdigit():
            return None
        return int(stdout)

    @staticmethod
    def _display(path: str, base: Path) -> str:
        candidate = Path(path)
        try:
            return candidate.relative_to(base).as_posix()
        except ValueError:
            return str(candidate)
