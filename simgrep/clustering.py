"""Deterministic semantic duplicate clustering over stored chunk vectors.

Pure computation: numpy in, :class:`~simgrep.models.SemanticCluster` out, no I/O.
`cluster_components` returns every qualifying cluster plus the pre-cap count
so callers can apply caps and build a
:class:`~simgrep.models.ClustersOutcome`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from simgrep.errors import ClustersError
from simgrep.models import ClusterMember, ClustersOptions, SemanticCluster

_BLOCK_ROWS = 4096
_MAX_CHUNKS_HINT = "Narrow the scope (e.g. a subdirectory) or raise --max-chunks."


class _UnionFind:
    """Union-find with path compression; roots keep the smallest member index."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, node: int) -> int:
        parent = self._parent
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self._parent[right_root] = left_root
        else:
            self._parent[left_root] = right_root


def _unioned_span_length(spans: list[tuple[int, int]]) -> int:
    """Total length of the union of inclusive ``[start, end]`` ranges."""
    total = 0
    span_end = None
    for start, end in sorted(spans):
        if span_end is not None and start <= span_end + 1:
            if end > span_end:
                total += end - span_end
                span_end = end
        else:
            total += end - start + 1
            span_end = end
    return total


def _duplicated_lines(members: list[ClusterMember]) -> int:
    spans_per_file: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for member in members:
        spans_per_file[str(Path(member.file_path))].append((member.line_start, member.line_end))
    return sum(_unioned_span_length(spans) for spans in spans_per_file.values())


def cluster_components(vectors: np.ndarray, members: Mapping[int, ClusterMember], options: ClustersOptions) -> tuple[list[SemanticCluster], int]:
    """Build every duplicate cluster passing ``min_size``, in final order.

    Returns the clusters ordered by ``(-duplicated_lines, -score, min label)``
    with members sorted by ``(file_path, line_start, label)``, plus the count of
    qualifying clusters before any ``top`` cap.
    """
    if options.min_size < 2:
        raise ClustersError(
            f"--min-size must be >= 2 (got {options.min_size}).",
            hint="A duplicate cluster needs at least two chunks; lower values are rejected.",
        )
    if not 0 < options.threshold <= 1:
        raise ClustersError(
            f"--threshold must satisfy 0 < threshold <= 1 (got {options.threshold}).",
            hint="Cosine similarity lies in [0, 1]; use a value like 0.8.",
        )
    if len(members) <= 1:
        return [], 0
    if len(members) > options.max_chunks:
        raise ClustersError(
            f"{len(members)} chunks exceed --max-chunks {options.max_chunks}.",
            hint=_MAX_CHUNKS_HINT,
        )

    labels = sorted(members)
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    kept = norms > 0.0  # zero-norm rows carry no direction: skip entirely
    kept_positions = np.flatnonzero(kept)
    normalized = matrix[kept] / norms[kept, None]
    count = len(kept_positions)
    file_id_of: dict[str, int] = {}
    file_ids = np.fromiter(
        (file_id_of.setdefault(str(Path(members[labels[pos]].file_path)), len(file_id_of)) for pos in kept_positions),
        dtype=np.int64,
        count=count,
    )

    union_find = _UnionFind(count)
    edges: list[tuple[int, int, float]] = []
    for block_start in range(0, count, _BLOCK_ROWS):
        block_stop = min(block_start + _BLOCK_ROWS, count)
        similarity = normalized[block_start:block_stop] @ normalized.T
        hit_rows, hit_cols = np.nonzero(similarity >= options.threshold)
        for block_row, col in zip(hit_rows.tolist(), hit_cols.tolist(), strict=True):
            row = block_start + block_row
            if col <= row:
                continue
            if not options.same_file and file_ids[row] == file_ids[col]:
                continue
            edges.append((row, col, float(similarity[block_row, col])))
            union_find.union(row, col)

    components: dict[int, list[int]] = defaultdict(list)
    for row in range(count):
        components[union_find.find(row)].append(row)

    min_edge_by_component: dict[int, float] = {}
    for row, col, score in edges:
        root = union_find.find(row)
        known = min_edge_by_component.get(root)
        if known is None or score < known:
            min_edge_by_component[root] = score

    clusters: list[SemanticCluster] = []
    for root, rows in components.items():
        if len(rows) < options.min_size:
            continue
        cluster_members = sorted(
            (members[labels[kept_positions[position]]] for position in rows),
            key=lambda m: (m.file_path, m.line_start, m.label),
        )
        clusters.append(
            SemanticCluster(
                members=tuple(cluster_members),
                score=min_edge_by_component[root],
                duplicated_lines=_duplicated_lines(cluster_members),
            )
        )

    clusters.sort(key=lambda c: (-c.duplicated_lines, -c.score, min(m.label for m in c.members)))
    return clusters, len(clusters)
