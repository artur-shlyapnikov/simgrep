"""Pure rerank math: orderings, windowed result reranking, best-per-file.

No I/O and no model access — callers supply a ``score_fn(query, documents)``
returning one cross score per document. Also hosts ``ensure_chunk_cap`` (the
standalone command's chunk budget guard) and ``chunk_file_texts`` (thin
pass-through to the runtime chunker).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from simgrep.errors import RerankError
from simgrep.models import RerankMatch, SearchOutcome, SearchResult


def ensure_chunk_cap(count: int, cap: int) -> None:
    """Raise :class:`RerankError` when ``count`` chunks exceed the ``cap``."""
    if count > cap:
        raise RerankError(
            f"{count} chunks exceed the {cap}-chunk rerank cap",
            hint="raise --max-chunks or narrow the file list",
        )


def rerank_orderings(cross_scores: Sequence[float]) -> tuple[int, ...]:
    """Argsort descending; ties keep original index order. Deterministic."""
    return tuple(sorted(range(len(cross_scores)), key=lambda i: (-cross_scores[i], i)))


def rerank_results(
    results: list[SearchResult],
    query: str,
    score_fn: Callable[[str, list[str]], "np.ndarray"],
    top: int,
) -> SearchOutcome:
    """Rerank the first ``min(top, len(results))`` entries by cross score.

    The window is scored via ``score_fn(query, [r.chunk_text ...])`` in incoming
    (hybrid) order, then sorted by ``(-cross, original_index)``; each windowed
    result's ``score`` field becomes its cross score. Results beyond the window
    are passed through untouched. Callers merge ``.results`` into their own
    outcome metadata (base_path, counters) — this fresh outcome carries none.
    """
    if not results:
        return SearchOutcome(results=[], base_path=Path("."))
    window_n = min(top, len(results))
    window = results[:window_n]
    cross = np.asarray(score_fn(query, [r.chunk_text for r in window]), dtype=np.float64)
    order = rerank_orderings(cross.tolist())
    reranked = [replace(window[i], score=float(cross[i])) for i in order]
    return SearchOutcome(results=reranked + list(results[window_n:]), base_path=Path("."))


def best_per_file(matches: Sequence[RerankMatch]) -> tuple[RerankMatch, ...]:
    """One match per file: max score wins, ties keep the lowest line_start.

    Output is sorted descending by score (line_start ascending on ties).
    """
    best: dict[str, RerankMatch] = {}
    for m in matches:
        current = best.get(m.file_path)
        if current is None or m.score > current.score or (m.score == current.score and m.line_start < current.line_start):
            best[m.file_path] = m
    return tuple(sorted(best.values(), key=lambda m: (-m.score, m.line_start)))


def chunk_file_texts(text: str, chunker: Any) -> list:
    """Thin delegation to ``chunker.chunk(text)``, returned as a plain list.

    Pure pass-through: chunk identity, offsets and line-start/line-end fields
    are preserved exactly as the chunker produced them — no re-computation here.
    """
    return list(chunker.chunk(text))
