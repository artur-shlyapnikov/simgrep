"""Unit tests for the pure clustering algorithm in simgrep/clustering.py."""

from __future__ import annotations

import numpy as np
import pytest

from simgrep.clustering import cluster_components
from simgrep.errors import ClustersError
from simgrep.models import ClusterMember, ClustersOptions, SemanticCluster

# 8-D directions: pairwise cosines of distinct families all stay below threshold 0.8,
# identical directions score exactly ~1.0, and cos(G, H) ~= 0.816 sits between.
DIR_A = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
DIR_B = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
DIR_G = (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
DIR_H = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)


def make_case(
    rows: list[tuple[int, str, int, int, tuple[float, ...]]],
) -> tuple[np.ndarray, dict[int, ClusterMember]]:
    ordered = sorted(rows, key=lambda row: row[0])  # vectors rows align with sorted(labels)
    vectors = np.array([row[4] for row in ordered], dtype=np.float32)
    members = {row[0]: ClusterMember(row[0], row[1], row[2], row[3]) for row in ordered}
    return vectors, members


def labels_of(cluster: SemanticCluster) -> tuple[int, ...]:
    return tuple(member.label for member in cluster.members)


def test_empty_members_returns_empty() -> None:
    vectors = np.zeros((0, 4), dtype=np.float32)
    assert cluster_components(vectors, {}, ClustersOptions()) == ([], 0)
    single = {7: ClusterMember(7, "a.py", 1, 2)}
    assert cluster_components(np.zeros((1, 4), dtype=np.float32), single, ClustersOptions()) == ([], 0)


def test_max_chunks_guard_raises_clusters_error() -> None:
    members = {i: ClusterMember(i, f"f{i}.py", 1, 2) for i in range(6)}
    with pytest.raises(ClustersError) as exc:
        cluster_components(np.zeros((6, 4), dtype=np.float32), members, ClustersOptions(max_chunks=5))
    assert "Narrow the scope" in (exc.value.hint or "")
    assert exc.value.exit_code == 1


def test_unnormalized_inputs_are_l2_normalized() -> None:
    # Same direction, wildly different magnitudes: must still cluster at 0.8.
    vectors, members = make_case(
        [
            (1, "a.py", 1, 10, DIR_A),
            (2, "b.py", 1, 5, tuple(50.0 * x for x in DIR_A)),
        ]
    )
    clusters, _ = cluster_components(vectors, members, ClustersOptions())
    assert len(clusters) == 1
    assert clusters[0].score == pytest.approx(1.0)


def test_zero_norm_rows_skipped_entirely() -> None:
    vectors, members = make_case(
        [
            (1, "a.py", 1, 10, DIR_A),
            (2, "b.py", 1, 5, (0.0,) * 8),
            (3, "c.py", 1, 5, DIR_A),
        ]
    )
    clusters, _ = cluster_components(vectors, members, ClustersOptions())
    assert len(clusters) == 1
    assert labels_of(clusters[0]) == (1, 3)


def test_threshold_boundary_is_inclusive() -> None:
    # Exactly representable unit vectors give cosine exactly 1.0 == threshold;
    # an unrelated orthogonal direction stays out.
    exact = (0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0)  # L2 norm exactly 1.0 in float32
    vectors, members = make_case(
        [
            (1, "a.py", 1, 2, DIR_A),
            (2, "b.py", 1, 2, exact),
            (3, "c.py", 1, 2, exact),
        ]
    )
    clusters, _ = cluster_components(vectors, members, ClustersOptions(threshold=1.0))
    assert len(clusters) == 1
    assert labels_of(clusters[0]) == (2, 3)
    assert clusters[0].score == pytest.approx(1.0)


def test_same_file_pair_excluded_by_default() -> None:
    vectors, members = make_case(
        [
            (1, "a.py", 1, 10, DIR_B),
            (2, "a.py", 20, 30, DIR_B),
        ]
    )
    assert cluster_components(vectors, members, ClustersOptions()) == ([], 0)


def test_same_file_pair_kept_when_same_file_enabled() -> None:
    vectors, members = make_case(
        [
            (1, "a.py", 1, 10, DIR_B),
            (2, "a.py", 20, 31, DIR_B),
        ]
    )
    clusters, _ = cluster_components(vectors, members, ClustersOptions(same_file=True))
    assert len(clusters) == 1
    assert clusters[0].duplicated_lines == 22  # union of [1,10] and [20,31]


def test_cross_file_happy_path_member_fields_and_sorting() -> None:
    vectors, members = make_case(
        [
            (9, "z.py", 5, 6, DIR_B),
            (4, "a.py", 100, 110, DIR_B),
        ]
    )
    clusters, _ = cluster_components(vectors, members, ClustersOptions())
    assert len(clusters) == 1
    cluster = clusters[0]
    assert labels_of(cluster) == (4, 9)  # sorted by (file_path, line_start, label)
    assert [(m.file_path, m.line_start, m.line_end) for m in cluster.members] == [
        ("a.py", 100, 110),
        ("z.py", 5, 6),
    ]
    assert cluster.duplicated_lines == 13
    assert cluster.score == pytest.approx(1.0)


def test_span_union_counts_overlapping_ranges_once() -> None:
    # a1 and a2 share file f1 (their mutual edge is dropped) but both link to b1 cross-file,
    # so all three land in one component; overlapping f1 spans must be counted once.
    vectors, members = make_case(
        [
            (1, "f1.py", 1, 10, DIR_A),
            (2, "f1.py", 6, 20, DIR_A),
            (3, "f2.py", 5, 7, DIR_A),
        ]
    )
    clusters, _ = cluster_components(vectors, members, ClustersOptions())
    assert len(clusters) == 1
    assert clusters[0].duplicated_lines == 23  # [1,20] union = 20, plus [5,7] = 3


def test_min_size_filters_small_components() -> None:
    vectors, members = make_case(
        [
            (1, "a.py", 1, 10, DIR_A),
            (2, "b.py", 1, 10, DIR_A),
            (3, "c.py", 1, 10, DIR_B),
            (4, "d.py", 1, 10, DIR_B),
        ]
    )
    pairs, _ = cluster_components(vectors, members, ClustersOptions(min_size=2))
    assert len(pairs) == 2
    assert cluster_components(vectors, members, ClustersOptions(min_size=3)) == ([], 0)


def test_top_cap_and_total_found() -> None:
    rows: list[tuple[int, str, int, int, tuple[float, ...]]] = []
    dirs = [DIR_A, DIR_B, (1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
    span_lengths = [10, 5, 2]
    label = 0
    for group, (direction, length) in enumerate(zip(dirs, span_lengths, strict=True)):
        for half in range(2):
            rows.append((label, f"g{group}_{half}.py", 1, length, direction))
            label += 1
    vectors, members = make_case(rows)
    options = ClustersOptions(top=2)
    clusters, total_found = cluster_components(vectors, members, options)
    assert total_found == 3
    assert [c.duplicated_lines for c in clusters[: options.top]] == [20, 10]
    assert [cluster.duplicated_lines for cluster in clusters] == [20, 10, 4]  # uncapped, ordered
    assert all(labels_of(c)[0] % 2 == 0 for c in clusters)


def test_ordering_tiebreak_by_score_desc_then_min_label() -> None:
    # Two clusters with equal duplicated_lines: higher score first; on equal scores,
    # smaller min label first.
    vectors, members = make_case(
        [
            (10, "x1.py", 1, 5, DIR_G),  # score ~0.816
            (11, "x2.py", 1, 5, DIR_H),
            (2, "y1.py", 1, 5, DIR_A),  # score exactly 1.0
            (3, "y2.py", 1, 5, DIR_A),
            (40, "w1.py", 1, 5, DIR_B),  # score exactly 1.0, min label > previous
            (41, "w2.py", 1, 5, DIR_B),
        ]
    )
    clusters, _ = cluster_components(vectors, members, ClustersOptions(threshold=0.8))
    assert [labels_of(c) for c in clusters] == [(2, 3), (40, 41), (10, 11)]
    assert clusters[2].score == pytest.approx(0.8164965, abs=1e-6)


def test_deterministic_two_runs_identical_tuples() -> None:
    rng = np.random.default_rng(42)
    base = rng.standard_normal((24, 8)).astype(np.float32)
    rows = []
    for i in range(24):
        direction = base[i % 8] if i < 16 else base[(i - 16) % 3]  # some near-threshold geometry
        scale = 1.0 + 0.001 * i
        rows.append((i, f"f{i % 6}.py", i * 10 + 1, i * 10 + 9, tuple(scale * direction)))
    vectors, members = make_case(rows)
    first = cluster_components(vectors, members, ClustersOptions(threshold=0.55))
    second = cluster_components(vectors, members, ClustersOptions(threshold=0.55))
    assert first == second
    assert len(first[0]) > 0


def test_large_input_blocked_path_matches_small_blocks() -> None:
    # Force multiple BLAS blocks (>4096 rows) and verify results stay correct.
    n = 4200
    members = {i: ClusterMember(i, f"f{i % 2}.py", i + 1, i + 2) for i in range(n)}
    unit = np.eye(2, dtype=np.float32)
    vectors = np.repeat(unit, n // 2, axis=0)  # alternating e0/e1 directions
    clusters, _ = cluster_components(vectors, members, ClustersOptions(same_file=True, top=1))
    clusters = clusters[:1]
    assert len(clusters[0].members) == n // 2


def test_ordering_tiebreak_uses_min_member_label_not_first_sorted_member() -> None:
    # Cluster X spans labels {1, 5}; its alphabetically-first member (a.py) holds
    # label 5, so members[0].label diverges from the true min label 1. With equal
    # duplicated_lines and scores against cluster Y {3, 4}, X must sort first.
    vectors, members = make_case(
        [
            (1, "z.py", 1, 2, DIR_A),
            (5, "a.py", 1, 2, DIR_A),
            (3, "b.py", 1, 2, DIR_B),
            (4, "c.py", 1, 2, DIR_B),
        ]
    )
    clusters, _ = cluster_components(vectors, members, ClustersOptions())
    assert len(clusters) == 2
    assert set(labels_of(clusters[0])) == {1, 5}
    assert set(labels_of(clusters[1])) == {3, 4}


def test_min_size_below_two_raises_clusters_error() -> None:
    # min_size=1 previously reached span merge and died with a cryptic KeyError.
    vectors, members = make_case(
        [
            (1, "a.py", 1, 2, DIR_A),
            (2, "b.py", 1, 2, DIR_A),
        ]
    )
    with pytest.raises(ClustersError) as exc:
        cluster_components(vectors, members, ClustersOptions(min_size=1))
    assert "--min-size" in str(exc.value)
    assert exc.value.hint is not None


def test_threshold_outside_zero_one_raises_clusters_error() -> None:
    vectors, members = make_case(
        [
            (1, "a.py", 1, 2, DIR_A),
            (2, "b.py", 1, 2, DIR_A),
        ]
    )
    for bad in (0.0, -0.5, 1.0001):
        with pytest.raises(ClustersError) as exc:
            cluster_components(vectors, members, ClustersOptions(threshold=bad))
        assert "--threshold" in str(exc.value)
        assert exc.value.hint is not None
