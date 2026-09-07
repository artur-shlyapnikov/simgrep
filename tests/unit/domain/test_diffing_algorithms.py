"""Unit tests for the pure tree-diff matching algorithm in simgrep/diffing.py."""

from __future__ import annotations

import numpy as np
import pytest

from simgrep import diffing
from simgrep.diffing import match_trees
from simgrep.errors import DiffError
from simgrep.models import DiffOptions

# 8-D orthogonal directions: cross-family cosines are ~0, identical ones are 1.0.
DIR_U = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
DIR_V = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
DIR_A = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
DIR_B = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
DIR_C = (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
DIR_D = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)


def matrix(rows: list[tuple[float, ...]]) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


def pair_set(pairs: list[tuple[int, int, float]]) -> set[tuple[int, int]]:
    return {(label_a, label_b) for label_a, label_b, _ in pairs}


def test_perfect_permutation_is_fully_recovered() -> None:
    # Tree B contains the same content in swapped row/label positions.
    vectors_a = matrix([DIR_A, DIR_B, DIR_C])
    vectors_b = matrix([DIR_C, DIR_A, DIR_B])
    pairs, unmatched_a, unmatched_b = match_trees(vectors_a, vectors_b, [10, 11, 12], [20, 21, 22], DiffOptions())
    assert pair_set(pairs) == {(10, 21), (11, 22), (12, 20)}
    assert unmatched_a == set()
    assert unmatched_b == set()
    for _, _, score in pairs:
        assert score == pytest.approx(1.0)

    # Unit basis vectors survive L2 normalization exactly, so an
    # identical-direction pair scores exactly 1.0; threshold 1.0 must match.
    pairs, unmatched_a, unmatched_b = match_trees(matrix([DIR_U, DIR_V]), matrix([DIR_U]), [1, 2], [9], DiffOptions(threshold=1.0))
    assert pair_set(pairs) == {(1, 9)}
    assert unmatched_a == {2}
    assert unmatched_b == set()


def test_below_threshold_never_matches() -> None:
    vectors_a = matrix([DIR_A])
    vectors_b = matrix([DIR_B])
    pairs, unmatched_a, unmatched_b = match_trees(vectors_a, vectors_b, [1], [2], DiffOptions(threshold=0.8))
    assert pairs == []
    assert unmatched_a == {1}
    assert unmatched_b == {2}


def test_greedy_tie_break_prefers_smaller_labels() -> None:
    # Both A chunks are equally similar to the single B chunk; the tie must be
    # broken deterministically by (label_a, label_b) i.e. the smaller label_a.
    vectors_a = matrix([DIR_A, DIR_A])
    vectors_b = matrix([DIR_A])
    pairs, unmatched_a, _ = match_trees(vectors_a, vectors_b, [7, 3], [5], DiffOptions())
    assert [(pair[0], pair[1]) for pair in pairs] == [(3, 5)]
    assert unmatched_a == {7}


def test_no_chunk_is_matched_twice() -> None:
    # Two identical A chunks face two identical B chunks plus one extra A chunk.
    vectors_a = matrix([DIR_A, DIR_A, DIR_A])
    vectors_b = matrix([DIR_A, DIR_A])
    pairs, unmatched_a, unmatched_b = match_trees(vectors_a, vectors_b, [1, 2, 3], [4, 5], DiffOptions())
    assert len(pairs) == 2
    assert {label_a for label_a, _, _ in pairs} == {1, 2}  # smaller labels win
    assert unmatched_a == {3}
    assert unmatched_b == set()
    seen_a = [label_a for label_a, _, _ in pairs]
    seen_b = [label_b for _, label_b, _ in pairs]
    assert len(seen_a) == len(set(seen_a))
    assert len(seen_b) == len(set(seen_b))


def test_empty_tree_a_returns_all_b_unmatched() -> None:
    pairs, unmatched_a, unmatched_b = match_trees(np.zeros((0, 8), dtype=np.float32), matrix([DIR_A, DIR_B]), [], [4, 5], DiffOptions())
    assert pairs == []
    assert unmatched_a == set()
    assert unmatched_b == {4, 5}


def test_empty_tree_b_returns_all_a_unmatched() -> None:
    pairs, unmatched_a, unmatched_b = match_trees(matrix([DIR_A, DIR_B]), np.zeros((0, 8), dtype=np.float32), [1, 2], [], DiffOptions())
    assert pairs == []
    assert unmatched_a == {1, 2}
    assert unmatched_b == set()


def test_zero_norm_rows_are_skipped() -> None:
    vectors_a = matrix([DIR_A, (0.0,) * 8, DIR_B])
    vectors_b = matrix([(0.0,) * 8, DIR_A, DIR_B])
    pairs, unmatched_a, unmatched_b = match_trees(vectors_a, vectors_b, [1, 2, 3], [8, 9, 10], DiffOptions())
    assert pair_set(pairs) == {(1, 9), (3, 10)}
    assert unmatched_a == {2}
    assert unmatched_b == {8}


def test_max_chunks_guard_raises_diff_error_with_pinned_hint() -> None:
    with pytest.raises(DiffError) as exc:
        match_trees(matrix([DIR_A]), matrix([DIR_A]), [1], [2], DiffOptions(max_chunks=1))
    assert exc.value.hint == "Narrow the scope (e.g. a subdirectory) or raise --max-chunks."
    assert exc.value.exit_code == 1


def test_shape_validation_rejects_mismatched_inputs() -> None:
    with pytest.raises(DiffError):
        match_trees(matrix([DIR_A]), matrix([DIR_A]), [1, 2], [2], DiffOptions())
    with pytest.raises(DiffError):
        match_trees(np.zeros((1, 7), dtype=np.float32), matrix([DIR_A]), [1], [2], DiffOptions())


def test_unnormalized_inputs_are_l2_normalized_internally() -> None:
    vectors_a = matrix([tuple(50.0 * x for x in DIR_A)])
    vectors_b = matrix([DIR_A, tuple(0.01 * x for x in DIR_A)])
    pairs, unmatched_a, unmatched_b = match_trees(vectors_a, vectors_b, [1], [2, 3], DiffOptions())
    # Both B rows share A's direction, so both are perfect matches after
    # normalization; only the tie-broken winner pairs up.
    assert len(pairs) == 1
    assert pairs[0][0] == 1
    assert pairs[0][1] == 2  # smaller label_b wins the tie
    assert pairs[0][2] == pytest.approx(1.0)
    assert unmatched_a == set()
    assert unmatched_b == {3}


def test_two_runs_are_identical() -> None:
    rng = np.random.default_rng(42)
    base = rng.standard_normal((40, 8)).astype(np.float32)
    vectors_a = base[:30]
    vectors_b = np.vstack([base[10:30], rng.standard_normal((5, 8))]).astype(np.float32)
    labels_a = list(range(100, 130))
    labels_b = list(range(200, 225))
    first = match_trees(vectors_a, vectors_b, labels_a, labels_b, DiffOptions(threshold=0.6))
    second = match_trees(vectors_a, vectors_b, labels_a, labels_b, DiffOptions(threshold=0.6))
    assert first == second


def test_blocked_matmul_path_matches_full_result(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(7)
    vectors_a = rng.standard_normal((25, 8)).astype(np.float32)
    vectors_b = rng.standard_normal((17, 8)).astype(np.float32)
    labels_a = list(range(1, 26))
    labels_b = list(range(101, 118))
    options = DiffOptions(threshold=0.05)
    reference = match_trees(vectors_a, vectors_b, labels_a, labels_b, options)
    monkeypatch.setattr(diffing, "_BLOCK_ROWS", 4)  # force many tiny blocks
    blocked = match_trees(vectors_a, vectors_b, labels_a, labels_b, options)
    # BLAS reduction order differs between block shapes, so compare the
    # pairing structure exactly and scores approximately.
    assert [(a, b) for a, b, _ in blocked[0]] == [(a, b) for a, b, _ in reference[0]]
    assert blocked[1] == reference[1]
    assert blocked[2] == reference[2]
    for (_, _, s_block), (_, _, s_ref) in zip(blocked[0], reference[0], strict=True):
        assert s_block == pytest.approx(s_ref, abs=1e-6)
