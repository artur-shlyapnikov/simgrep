"""Round-16 property coverage for the pure matching core (simgrep/diffing.py).

Pins:
- mid-range threshold-boundary inclusivity (recomputed through production's fp32 op chain);
- maximal-matching invariant: no unmatched cross pair qualifies under the threshold;
- threshold monotonicity with bit-identical shared scores;
- B-side row-permutation invariance (structure exact, scores approx);
- scores equal independently recomputed cosines, returned in descending assignment order;
- B-side tie-break by label, not row position;
- exact validation messages + hints for all four DiffError branches;
- max_chunks == boundary and guard-before-empty-short-circuit precedence.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from simgrep.diffing import match_trees
from simgrep.errors import DiffError
from simgrep.models import DiffOptions

DIR_A = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def matrix(rows: list[tuple[float, ...]]) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


def _cosine_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Oracle cosine matrix over the NONZERO-norm rows of each side."""
    kept_x = x[np.linalg.norm(x, axis=1) > 0.0]
    kept_y = y[np.linalg.norm(y, axis=1) > 0.0]
    xn = kept_x / np.linalg.norm(kept_x, axis=1)[:, None]
    yn = kept_y / np.linalg.norm(kept_y, axis=1)[:, None]
    product: np.ndarray = xn @ yn.T
    return product


def test_threshold_boundary_is_inclusive() -> None:
    a = matrix([(3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)])
    b = matrix([(4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)])
    na = a / np.linalg.norm(a, axis=1, keepdims=True)
    nb = b / np.linalg.norm(b, axis=1, keepdims=True)
    sim = float((na @ nb.T)[0, 0])

    pairs, unmatched_a, unmatched_b = match_trees(a, b, [1], [2], DiffOptions(threshold=sim))
    assert pairs == [(1, 2, sim)]
    assert unmatched_a == set()
    assert unmatched_b == set()

    strict = float(np.nextafter(np.float32(sim), np.float32(1.0)))
    pairs_strict, unmatched_a_strict, unmatched_b_strict = match_trees(a, b, [1], [2], DiffOptions(threshold=strict))
    assert pairs_strict == []
    assert unmatched_a_strict == {1}
    assert unmatched_b_strict == {2}


_SHAPE_A = st.integers(min_value=1, max_value=24)
_SHAPE_B = st.integers(min_value=1, max_value=24)
_SEED = st.integers(min_value=0, max_value=2**32 - 1)


def _random_side(rng: np.random.Generator, rows: int) -> np.ndarray:
    vectors = rng.standard_normal((rows, 6)).astype(np.float32)
    if rows >= 2:
        vectors[0] = 0.0  # at most one zero row per side; it must land unmatched
    return vectors


@settings(max_examples=50, deadline=None)
@given(n_a=_SHAPE_A, n_b=_SHAPE_B, seed=_SEED)
def test_no_qualifying_cross_pair_survives_unmatched(n_a: int, n_b: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    vectors_a = _random_side(rng, n_a)
    vectors_b = _random_side(rng, n_b)
    labels_a = list(range(100, 100 + n_a))
    labels_b = list(range(500, 500 + n_b))
    pairs, unmatched_a, unmatched_b = match_trees(vectors_a, vectors_b, labels_a, labels_b, DiffOptions(threshold=0.35))
    cosines = _cosine_matrix(vectors_a, vectors_b)
    pos_a = np.flatnonzero(np.linalg.norm(vectors_a, axis=1) > 0.0)
    pos_b = np.flatnonzero(np.linalg.norm(vectors_b, axis=1) > 0.0)
    row_of_label_a = {labels_a[int(pos)]: row for row, pos in enumerate(pos_a.tolist())}
    row_of_label_b = {labels_b[int(pos)]: row for row, pos in enumerate(pos_b.tolist())}
    for label_a in unmatched_a:
        for label_b in unmatched_b:
            if label_a not in row_of_label_a or label_b not in row_of_label_b:
                continue  # zero-norm rows carry no direction and cannot qualify
            assert cosines[row_of_label_a[label_a], row_of_label_b[label_b]] < 0.35


@settings(max_examples=50, deadline=None)
@given(n_a=_SHAPE_A, n_b=_SHAPE_B, seed=_SEED)
def test_pairs_are_monotone_in_threshold(n_a: int, n_b: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    vectors_a = _random_side(rng, n_a)
    vectors_b = _random_side(rng, n_b)
    labels_a = list(range(100, 100 + n_a))
    labels_b = list(range(500, 500 + n_b))
    pairs_hi, _, _ = match_trees(vectors_a, vectors_b, labels_a, labels_b, DiffOptions(threshold=0.7))
    pairs_lo, _, _ = match_trees(vectors_a, vectors_b, labels_a, labels_b, DiffOptions(threshold=0.35))
    set_hi = {(x, y) for x, y, _ in pairs_hi}
    set_lo = {(x, y) for x, y, _ in pairs_lo}
    assert set_hi <= set_lo
    score_lo = {(x, y): s for x, y, s in pairs_lo}
    for x, y, score in pairs_hi:
        # scores are computed before the threshold filter, so shared pairs are bit-equal
        assert score_lo[(x, y)] == score


@settings(max_examples=50, deadline=None)
@given(n_a=_SHAPE_A, n_b=_SHAPE_B, seed=_SEED)
def test_b_side_permutation_preserves_matching_structure(n_a: int, n_b: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    vectors_a = _random_side(rng, n_a)
    vectors_b = _random_side(rng, n_b)
    labels_a = list(range(100, 100 + n_a))
    labels_b = list(range(500, 500 + n_b))
    base_pairs, base_unmatched_a, base_unmatched_b = match_trees(vectors_a, vectors_b, labels_a, labels_b, DiffOptions(threshold=0.35))

    perm = rng.permutation(n_b)
    permuted_vectors_b = vectors_b[perm]
    permuted_labels_b = [labels_b[i] for i in perm.tolist()]
    pairs, unmatched_a, unmatched_b = match_trees(vectors_a, permuted_vectors_b, labels_a, permuted_labels_b, DiffOptions(threshold=0.35))

    base_scores = {(x, y): s for x, y, s in base_pairs}
    assert {(x, y) for x, y, _ in pairs} == {(x, y) for x, y, _ in base_pairs}
    for x, y, score in pairs:
        assert score == pytest.approx(base_scores[(x, y)], abs=1e-6)
    assert unmatched_a == base_unmatched_a
    assert unmatched_b == base_unmatched_b


def test_scores_equal_recomputed_cosines_and_come_in_assignment_order() -> None:
    rng = np.random.default_rng(11)
    vectors_a = rng.standard_normal((14, 8)).astype(np.float32)
    vectors_b = rng.standard_normal((10, 8)).astype(np.float32)
    labels_a = list(range(100, 114))
    labels_b = list(range(500, 510))
    pairs, unmatched_a, unmatched_b = match_trees(vectors_a, vectors_b, labels_a, labels_b, DiffOptions(threshold=0.25))

    scores = [s for _, _, s in pairs]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    normalized_a = vectors_a / np.linalg.norm(vectors_a, axis=1, keepdims=True)
    normalized_b = vectors_b / np.linalg.norm(vectors_b, axis=1, keepdims=True)
    row_of_label_a = {label: row for row, label in enumerate(labels_a)}
    row_of_label_b = {label: row for row, label in enumerate(labels_b)}
    for label_a, label_b, score in pairs:
        assert score == pytest.approx(float(normalized_a[row_of_label_a[label_a]] @ normalized_b[row_of_label_b[label_b]]), abs=1e-6)

    matched_a = {label_a for label_a, _, _ in pairs}
    matched_b = {label_b for _, label_b, _ in pairs}
    assert matched_a.isdisjoint(unmatched_a) and matched_a | unmatched_a == set(labels_a)
    assert matched_b.isdisjoint(unmatched_b) and matched_b | unmatched_b == set(labels_b)


def test_tie_break_prefers_smaller_label_b_regardless_of_row_order() -> None:
    vectors_a = matrix([DIR_A])
    vectors_b = matrix([DIR_A, DIR_A])
    pairs, unmatched_a, unmatched_b = match_trees(vectors_a, vectors_b, [7], [9, 4], DiffOptions())
    assert pairs == [(7, 4, 1.0)]
    assert unmatched_a == set()
    assert unmatched_b == {9}


_VALIDATION_CASES: list[tuple[str, DiffOptions, Callable[[DiffOptions], object], str, str]] = [
    (
        "ndim",
        DiffOptions(),
        lambda opts: match_trees(np.zeros(4, dtype=np.float32), matrix([DIR_A]), [1], [2], opts),
        "vectors_a and vectors_b must be 2-D (N, dim) matrices.",
        "Both sides must provide one embedding row per chunk label.",
    ),
    (
        "dim",
        DiffOptions(),
        lambda opts: match_trees(np.zeros((1, 7), dtype=np.float32), matrix([DIR_A]), [1], [2], opts),
        "Embedding dimensions differ across trees (7 vs 8).",
        "Both trees must be embedded with the same model.",
    ),
    (
        "labels",
        DiffOptions(),
        lambda opts: match_trees(matrix([DIR_A]), matrix([DIR_A]), [1, 2], [2], opts),
        "Vector/label mismatch: A has 1 rows vs 2 labels, B has 1 rows vs 1 labels.",
        "Each chunk label needs exactly one vector row.",
    ),
    (
        "max_chunks",
        DiffOptions(max_chunks=1),
        lambda opts: match_trees(matrix([DIR_A]), matrix([DIR_A]), [1], [2], opts),
        "2 chunks exceed --max-chunks 1.",
        "Narrow the scope (e.g. a subdirectory) or raise --max-chunks.",
    ),
]


@pytest.mark.parametrize(
    ("name", "options", "run", "message", "hint"),
    _VALIDATION_CASES,
    ids=[case[0] for case in _VALIDATION_CASES],
)
def test_validation_errors_pin_exact_messages_and_hints(name: str, options: DiffOptions, run: Callable[[DiffOptions], object], message: str, hint: str) -> None:
    del name
    with pytest.raises(DiffError) as exc:
        run(options)
    assert str(exc.value) == message
    assert exc.value.hint == hint


def test_max_chunks_boundary_passes_and_guards_before_empty_short_circuit() -> None:
    pairs, unmatched_a, unmatched_b = match_trees(matrix([DIR_A]), matrix([DIR_A]), [1], [2], DiffOptions(max_chunks=2))
    assert pairs == [(1, 2, 1.0)]
    assert unmatched_a == set()
    assert unmatched_b == set()

    with pytest.raises(DiffError) as exc:
        match_trees(np.zeros((0, 8), dtype=np.float32), matrix([DIR_A]), [], [2], DiffOptions(max_chunks=0))
    assert str(exc.value) == "1 chunks exceed --max-chunks 0."
    assert exc.value.hint == "Narrow the scope (e.g. a subdirectory) or raise --max-chunks."
