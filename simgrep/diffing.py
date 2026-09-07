"""Deterministic greedy one-to-one matching for the ``simgrep diff`` feature.

Pure computation: two vector matrices plus label lists in, matched pairs and
unmatched label sets out — no I/O. The algorithm is fully deterministic:
candidates are ordered by ``(-score, label_a, label_b)`` before greedy
assignment, so identical inputs always yield identical outputs.
"""

from __future__ import annotations

import numpy as np

from simgrep.errors import DiffError
from simgrep.models import DiffOptions

_BLOCK_ROWS = 4096
_MAX_CHUNKS_HINT = "Narrow the scope (e.g. a subdirectory) or raise --max-chunks."


def match_trees(
    vectors_a: np.ndarray,
    vectors_b: np.ndarray,
    labels_a: list[int],
    labels_b: list[int],
    options: DiffOptions,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """Greedy one-to-one threshold matching.

    Returns ``(pairs of (label_a, label_b, score), unmatched_a_labels,
    unmatched_b_labels)``. Pairs come back in assignment order (best score
    first); the unmatched sets carry every label that ended up without a
    partner.
    """
    matrix_a = np.asarray(vectors_a)
    matrix_b = np.asarray(vectors_b)
    if matrix_a.ndim != 2 or matrix_b.ndim != 2:
        raise DiffError(
            "vectors_a and vectors_b must be 2-D (N, dim) matrices.",
            hint="Both sides must provide one embedding row per chunk label.",
        )
    count_a, dim_a = matrix_a.shape
    count_b, dim_b = matrix_b.shape
    if dim_a != dim_b:
        raise DiffError(
            f"Embedding dimensions differ across trees ({dim_a} vs {dim_b}).",
            hint="Both trees must be embedded with the same model.",
        )
    if count_a != len(labels_a) or count_b != len(labels_b):
        raise DiffError(
            f"Vector/label mismatch: A has {count_a} rows vs {len(labels_a)} labels, " f"B has {count_b} rows vs {len(labels_b)} labels.",
            hint="Each chunk label needs exactly one vector row.",
        )
    if count_a + count_b > options.max_chunks:
        raise DiffError(
            f"{count_a + count_b} chunks exceed --max-chunks {options.max_chunks}.",
            hint=_MAX_CHUNKS_HINT,
        )
    if count_a == 0 or count_b == 0:
        return [], set(labels_a), set(labels_b)

    norms_a = np.linalg.norm(matrix_a, axis=1)
    kept_a = norms_a > 0.0  # zero-norm rows carry no direction: skip entirely
    positions_a = np.flatnonzero(kept_a)
    normalized_a = matrix_a[kept_a] / norms_a[kept_a, None]
    labels_kept_a = [labels_a[pos] for pos in positions_a.tolist()]

    norms_b = np.linalg.norm(matrix_b, axis=1)
    kept_b = norms_b > 0.0
    positions_b = np.flatnonzero(kept_b)
    normalized_b = matrix_b[kept_b] / norms_b[kept_b, None]
    labels_kept_b = [labels_b[pos] for pos in positions_b.tolist()]

    candidates: list[tuple[float, int, int]] = []
    for block_start in range(0, len(labels_kept_a), _BLOCK_ROWS):
        block_stop = min(block_start + _BLOCK_ROWS, len(labels_kept_a))
        similarity = normalized_a[block_start:block_stop] @ normalized_b.T
        hit_rows, hit_cols = np.nonzero(similarity >= options.threshold)
        for block_row, col in zip(hit_rows.tolist(), hit_cols.tolist(), strict=True):
            candidates.append(
                (
                    float(similarity[block_row, col]),
                    labels_kept_a[block_start + block_row],
                    labels_kept_b[col],
                )
            )

    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))
    pairs: list[tuple[int, int, float]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    for score, label_a, label_b in candidates:
        if label_a in used_a or label_b in used_b:
            continue
        used_a.add(label_a)
        used_b.add(label_b)
        pairs.append((label_a, label_b, score))

    return pairs, set(labels_a) - used_a, set(labels_b) - used_b
