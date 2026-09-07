"""Pure math for ``simgrep debt``: marker scanning, semantic clustering, labels.

No imports from engines/adapters — numpy and the standard library only. The
engine (``simgrep.debt_engine``) feeds chunk rows into :func:`build_report`.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from simgrep.models import DebtMatch, DebtOptions, DebtReport, DebtTheme

MARKERS: tuple[str, ...] = ("TODO", "FIXME", "XXX", "HACK", "WORKAROUND")

# Case-SENSITIVE on purpose: uppercase is the convention these markers follow.
_MARKER_RE = re.compile(r"\b(TODO|FIXME|XXX|HACK|WORKAROUND)\b")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "on",
        "is",
        "are",
        "be",
        "this",
        "that",
        "with",
        "as",
        "at",
        "it",
        "we",
        "not",
        "but",
        "if",
        "when",
        "from",
        "by",
        "should",
        "must",
        "need",
        "needs",
    }
)

_MARKER_WORDS: frozenset[str] = frozenset(marker.lower() for marker in MARKERS)

_TOKEN_RE = re.compile(r"[a-z_][a-z0-9_]{2,}")

_SNIPPET_CAP = 120  # 117 chars + "..."
DAY_SECONDS = 86_400

_LABEL_FALLBACK = "debt"


def scan_text(text: str, line_start: int) -> list[tuple[int, str, str]]:
    """(absolute_line, marker, snippet) per marker occurrence; line = line_start + offset."""
    hits: list[tuple[int, str, str]] = []
    for offset, line in enumerate(text.splitlines()):
        for match in _MARKER_RE.finditer(line):
            snippet = line[match.end() :].strip()
            if not snippet:
                snippet = line.strip()  # bare marker line: show the whole line
            if len(snippet) > _SNIPPET_CAP - 3:
                snippet = snippet[: _SNIPPET_CAP - 3] + "..."
            hits.append((line_start + offset, match.group(1), snippet))
    return hits


def cluster_rows(vectors: np.ndarray, threshold: float) -> list[list[int]]:
    """Union-find over pairwise cosine >= threshold (L2-normalized rows, blocked matmul,
    zero-norm rows never join). Returns row-index components, each sorted asc, list sorted
    by (-size, first_index). Deterministic."""
    count = int(vectors.shape[0])
    parent = list(range(count))

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    norms = np.linalg.norm(vectors, axis=1)
    live = norms > 0.0
    safe = np.where(live, norms, 1.0)
    normalized = vectors.astype(np.float32, copy=False) / safe[:, np.newaxis].astype(np.float32)

    block = 512
    for start in range(0, count, block):
        end = min(start + block, count)
        if not live[start:end].any():
            continue
        sims = normalized @ normalized[start:end].T  # (count, end-start)
        for local in range(end - start):
            row = start + local
            if not live[row]:
                continue
            column = np.where(live, sims[:, local], -1.0)
            for other in np.nonzero(column >= threshold)[0]:
                union(row, int(other))

    members: dict[int, list[int]] = {}
    for row in range(count):
        members.setdefault(find(row), []).append(row)
    components = [sorted(rows) for rows in members.values()]
    return sorted(components, key=lambda comp: (-len(comp), comp[0]))


def theme_label(member_texts: Sequence[str]) -> str:
    """Top-2 tokens by (count desc, token asc) from [a-z_][a-z0-9_]{2,} lowercased tokens,
    minus _STOPWORDS and marker words; ' / '-joined; fallback 'debt'."""
    counts: Counter[str] = Counter()
    for text in member_texts:
        for token in _TOKEN_RE.findall(text.lower()):
            if token in _STOPWORDS or token in _MARKER_WORDS:
                continue
            counts[token] += 1
    if not counts:
        return _LABEL_FALLBACK
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:2]
    return " / ".join(token for token, _ in top)


def build_report(
    candidates: Sequence[tuple[int, str, int, str, str]],  # (row_idx, display_path, line, marker, snippet)
    vectors: np.ndarray,  # aligned to row_idx order
    texts_by_row: Mapping[int, str],
    epochs_by_path: Mapping[str, int | None],
    options: DebtOptions,
) -> DebtReport:
    """Cluster -> themes (size >= min_size) -> labels -> oldest_epoch -> rank -> cap top
    -> scattered/markers_found/truncated -> passed."""
    components = cluster_rows(vectors, options.threshold)

    themes: list[DebtTheme] = []
    occurrences_by_row: Counter[int] = Counter(candidate[0] for candidate in candidates)
    themed_markers = 0
    for component in components:
        if len(component) < options.min_size:
            continue
        themed_markers += sum(occurrences_by_row[row] for row in component)
        component_rows = set(component)
        members = [candidate for candidate in candidates if candidate[0] in component_rows]
        matches = sorted(
            (DebtMatch(file_path=path, line_start=line, marker=marker, snippet=snippet) for _, path, line, marker, snippet in members),
            key=lambda match: (match.file_path, match.line_start),
        )[: options.max_members]
        paths = {path for _, path, *_ in members}
        known = [epoch for epoch in (epochs_by_path.get(path) for path in paths) if epoch is not None]
        themes.append(
            DebtTheme(
                label=theme_label([texts_by_row[row] for row in component]),
                size=len(component),
                matches=tuple(matches),
                oldest_epoch=min(known) if known else None,
            )
        )

    themes.sort(key=lambda theme: (-theme.size, theme.oldest_epoch is None, theme.oldest_epoch or 0, theme.label))
    truncated = len(themes) > options.top
    passed: bool | None = None
    if options.max_age_days is not None:
        now = time.time()
        passed = not any(theme.oldest_epoch is not None and (now - theme.oldest_epoch) / DAY_SECONDS > options.max_age_days for theme in themes)
    return DebtReport(
        themes=tuple(themes[: options.top]),
        scattered=len(candidates) - themed_markers,
        markers_found=len(candidates),
        chunks_scanned=len(vectors),
        truncated=truncated,
        threshold=options.threshold,
        max_age_days=options.max_age_days,
        passed=passed,
    )
