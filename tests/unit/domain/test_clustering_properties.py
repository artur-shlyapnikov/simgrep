"""Round-13 property coverage for the pure clustering core (simgrep/clustering.py).

Pins:
- _unioned_span_length equals the covered-integer-line-set size for well-formed
  spans (independent oracle; subsumes adjacency merging, nesting, unsorted input);
- _duplicated_lines unions spans per file only -- cross-file overlaps add up;
- output is invariant to members-dict insertion order and to uniform label
  offsets (only label ORDER matters);
- chained components merge transitively and score by the weakest QUALIFYING edge,
  not the weakest pairwise similarity inside the component;
- relocated product bug pin (negative ``top``) lives in
  tests/unit/application/test_clusters_engine_edges.py since 0fc43df removed
  ``build_clusters``; the bare top slice now lives in ClustersEngine.run_handle.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from simgrep.clustering import (
    _duplicated_lines,
    _unioned_span_length,
    cluster_components,
)
from simgrep.models import ClusterMember, ClustersOptions


def _member(label: int, file_path: str, line_start: int, line_end: int) -> ClusterMember:
    return ClusterMember(label=label, file_path=file_path, line_start=line_start, line_end=line_end)


_LINE = st.integers(min_value=1, max_value=60)
_SPAN = st.tuples(_LINE, _LINE).filter(lambda s: s[0] <= s[1])
_FILE_NAME = st.sampled_from(["a.py", "b.py", "c.py"])
_SPANS = st.lists(st.tuples(_FILE_NAME, _SPAN), min_size=1, max_size=12)


@settings(max_examples=75, deadline=None)
@given(spans=_SPANS)
def test_unioned_span_length_equals_covered_line_set(spans: list[tuple[str, tuple[int, int]]]) -> None:
    """Oracle: for start<=end spans the sorted sweep == size of the covered lines."""
    ranges = [span for _, span in spans]
    expected = len({line for start, end in ranges for line in range(start, end + 1)})
    assert _unioned_span_length(ranges) == expected


@settings(max_examples=75, deadline=None)
@given(spans=_SPANS)
def test_duplicated_lines_unions_spans_per_file_only(
    spans: list[tuple[str, tuple[int, int]]],
) -> None:
    """Cross-file overlaps must add up; only same-file spans share a union."""
    members = [_member(label=i, file_path=name, line_start=start, line_end=end) for i, (name, (start, end)) in enumerate(spans, start=1)]
    expected = sum(len({line for name2, (start, end) in spans if name2 == name for line in range(start, end + 1)}) for name in {name for name, _ in spans})
    assert _duplicated_lines(members) == expected


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1), offset=st.integers(min_value=0, max_value=500))
def test_cluster_components_invariant_under_dict_order_and_label_offset(seed: int, offset: int) -> None:
    """Output depends on label ORDER, never on dict insertion order or label VALUES:
    a uniformly shifted labeling must yield the identical cluster structure."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 14))
    base = rng.standard_normal((n, 6)).astype(np.float32)
    base[n // 2] = base[0] * 3.0  # guarantee at least one duplicate direction
    labels = [(i + 1) * 7 + offset for i in range(n)]  # gapped, order-preserving labels
    members = {lab: _member(lab, f"f{i % 3}.py", i + 1, i + 2) for i, lab in enumerate(labels)}
    options = ClustersOptions(same_file=True)
    canonical, _ = cluster_components(base, members, options)

    shuffled_members = {labels[j]: members[labels[j]] for j in rng.permutation(n)}
    shifted_members = {lab + 100000: _member(lab + 100000, m.file_path, m.line_start, m.line_end) for lab, m in members.items()}

    def structure(cs: list, shift: int) -> list:
        return [([m.label - shift for m in c.members], round(c.score, 6), c.duplicated_lines) for c in cs]

    assert structure(cluster_components(base, shuffled_members, options)[0], 0) == structure(canonical, 0)
    assert structure(cluster_components(base, shifted_members, options)[0], 100000) == structure(canonical, 0)


def _pair_at(cosine: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([cosine, math.sqrt(1.0 - cosine * cosine)], dtype=np.float32)
    return a, b


def test_chain_forms_one_component_scored_by_weakest_qualifying_edge() -> None:
    """A->B at ~0.97 and B->C at ~0.90 chain into ONE component even though
    cos(A,C) ~0.77 sits below the threshold; score is the weakest QUALIFYING
    edge (0.90), not the weakest pairwise similarity in the component."""
    v1, v2 = _pair_at(0.97)
    angle = math.acos(0.97) + math.acos(0.90)
    v3 = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
    vectors = np.stack([v1, v2, v3]).astype(np.float32)
    members = {
        1: _member(1, "a.py", 1, 1),
        2: _member(2, "b.py", 1, 1),
        3: _member(3, "c.py", 1, 1),
    }

    clusters, total_found = cluster_components(vectors, members, ClustersOptions(threshold=0.85))

    assert total_found == 1
    assert [m.label for m in clusters[0].members] == [1, 2, 3]
    assert clusters[0].score == pytest.approx(0.90, abs=1e-5)
