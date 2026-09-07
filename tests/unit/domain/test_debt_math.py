"""Unit tests for simgrep.debt pure math (scan, cluster, label, report)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from simgrep.debt import build_report, cluster_rows, scan_text, theme_label
from simgrep.models import DebtMatch, DebtOptions, DebtReport, DebtTheme

DAY = 86_400


# ---------------------------------------------------------------- scan_text


def test_scan_text_finds_markers_with_line_offsets() -> None:
    text = "x = 1  # TODO fix later\ny = 2\n    # FIXME broken\n"
    assert scan_text(text, 10) == [
        (10, "TODO", "fix later"),
        (12, "FIXME", "broken"),
    ]


def test_scan_text_respects_word_boundaries_and_case() -> None:
    assert scan_text("TODOS plural stay\n", 1) == []
    assert scan_text("todo lowercase stays\n", 1) == []
    hits = scan_text("# TODO: tighten\n", 3)
    # rest-of-line after the marker, whitespace-stripped
    assert hits == [(3, "TODO", ": tighten")]


def test_scan_text_caps_snippet_at_117_plus_ellipsis() -> None:
    rest = "x" * 300
    hits = scan_text(f"# TODO {rest}\n", 1)
    assert len(hits) == 1
    _, marker, snippet = hits[0]
    assert marker == "TODO"
    assert snippet == "x" * 117 + "..."
    assert len(snippet) == 120


def test_scan_text_multiple_markers_on_one_line() -> None:
    hits = scan_text("# TODO a FIXME b\n", 5)
    assert hits == [(5, "TODO", "a FIXME b"), (5, "FIXME", "b")]


# ------------------------------------------------------------- cluster_rows


def test_cluster_rows_two_clusters_and_singleton_sorted_by_size_then_first_index() -> None:
    vectors = np.array(
        [
            [2.0, 0.0],  # A
            [2.0, 0.5],  # A (cos ~0.97 with row 0)
            [0.0, 3.0],  # B singleton direction
            [0.0, 0.0],  # zero-norm: never joins
        ],
        dtype=np.float32,
    )
    assert cluster_rows(vectors, threshold=0.8) == [[0, 1], [2], [3]]


def test_cluster_rows_threshold_boundary_separates_near_pairs() -> None:
    high = np.array([[1.0, 0.0], [0.81, 0.586_308]], dtype=np.float32)  # cos ~0.81
    low = np.array([[1.0, 0.0], [0.79, 0.613_049]], dtype=np.float32)  # cos ~0.79
    joined = cluster_rows(high, threshold=0.8)
    split = cluster_rows(low, threshold=0.8)
    assert joined == [[0, 1]]
    assert split == [[0], [1]]


def test_cluster_rows_zero_norm_rows_never_join_even_at_threshold_zero() -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    assert cluster_rows(vectors, threshold=0.0) == [[0, 1], [2]]


def test_cluster_rows_is_deterministic_across_runs() -> None:
    rng = np.random.default_rng(7)
    base = rng.normal(size=(12, 4)).astype(np.float32)
    base[3] = base[0] * 2.0  # force some joins
    base[7] = base[5] + 0.01
    first = cluster_rows(base, threshold=0.95)
    second = cluster_rows(base, threshold=0.95)
    assert first == second
    for component in first:
        assert component == sorted(component)


# -------------------------------------------------------------- theme_label


def test_theme_label_drops_stopwords_and_tie_breaks_token_asc() -> None:
    assert theme_label(["the parser config and retry"]) == "config / parser"


def test_theme_label_prefers_higher_counts_then_token_asc() -> None:
    assert theme_label(["retry retry backoff"]) == "retry / backoff"


def test_theme_label_excludes_marker_words() -> None:
    assert theme_label(["TODO fix the workaround logic here"]) == "fix / here"


def test_theme_label_falls_back_to_debt() -> None:
    assert theme_label(["we must not do this!"]) == "debt"
    assert theme_label([]) == "debt"


# ------------------------------------------------------------- build_report


def _candidates() -> list[tuple[int, str, int, str, str]]:
    return [
        (0, "a.py", 1, "TODO", "x"),
        (1, "a.py", 5, "FIXME", "y"),
        (2, "b.py", 2, "HACK", "z"),
        (3, "c.py", 7, "XXX", "w"),
    ]


def _vectors_two_themes() -> np.ndarray:
    return np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )


def _texts() -> dict[int, str]:
    return {0: "alpha parser", 1: "parser beta", 2: "gamma auth", 3: "auth delta"}


def test_build_report_ranks_caps_and_counts() -> None:
    now = time.time()
    epochs = {"a.py": int(now - 400 * DAY), "b.py": int(now - 10 * DAY), "c.py": None}
    options = DebtOptions(threshold=0.8, min_size=2, top=1, max_members=1)
    report = build_report(_candidates(), _vectors_two_themes(), _texts(), epochs, options)

    assert report.markers_found == 4
    assert report.chunks_scanned == 4
    assert report.scattered == 0
    assert report.truncated is True
    assert len(report.themes) == 1
    assert report.passed is None

    top = report.themes[0]
    assert top.label == "parser / alpha"  # count desc (parser=2), then token asc
    # equal sizes -> older epoch wins the top slot
    assert top.size == 2
    assert top.oldest_epoch == epochs["a.py"]
    # matches sorted by (file_path, line_start), capped at max_members
    assert top.matches == (DebtMatch("a.py", 1, "TODO", "x"),)


def test_build_report_none_epoch_ranks_last_within_equal_sizes() -> None:
    now = time.time()
    epochs = {"a.py": int(now - 400 * DAY), "b.py": int(now - 10 * DAY), "c.py": None}
    options = DebtOptions(threshold=0.8, min_size=2, top=5, max_members=8)
    report = build_report(_candidates(), _vectors_two_themes(), _texts(), epochs, options)

    assert report.truncated is False
    assert len(report.themes) == 2
    assert [theme.oldest_epoch for theme in report.themes] == [epochs["a.py"], epochs["b.py"]]


def test_build_report_counts_members_of_sub_min_size_components_as_scattered() -> None:
    candidates = [(0, "a.py", 1, "TODO", "x"), (1, "b.py", 1, "TODO", "y")]
    vectors = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    options = DebtOptions(threshold=0.8, min_size=3)
    report = build_report(candidates, vectors, {0: "alpha", 1: "alpha"}, {}, options)
    assert report.themes == ()
    assert report.scattered == 2
    assert report.markers_found == 2


def test_build_report_passed_gate_matrix() -> None:
    vectors = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    texts = {0: "retry logic", 1: "more retry"}
    candidates = [(0, "old.py", 1, "FIXME", "x"), (1, "new.py", 1, "FIXME", "y")]

    old_epoch = int(time.time() - 400 * DAY)
    recent_epoch = int(time.time() - 10 * DAY)

    def run(epochs: dict[str, int | None], max_age_days: float | None) -> DebtReport:
        options = DebtOptions(threshold=0.8, min_size=2, max_age_days=max_age_days)
        return build_report(candidates, vectors, texts, epochs, options)

    # no gate -> None regardless of ages
    assert run({"old.py": old_epoch, "new.py": old_epoch}, None).passed is None
    # dated theme within the gate -> True
    assert run({"old.py": recent_epoch, "new.py": recent_epoch}, 90.0).passed is True
    # dated theme older than the gate -> False
    assert run({"old.py": old_epoch, "new.py": old_epoch}, 90.0).passed is False
    # undated themes never fail the gate
    assert run({"old.py": None, "new.py": None}, 90.0).passed is True
    # mixed: oldest epoch drives the verdict
    assert run({"old.py": old_epoch, "new.py": recent_epoch}, 90.0).passed is False


def test_build_report_theme_carries_tuple_types() -> None:
    vectors = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    options = DebtOptions(threshold=0.8, min_size=2)
    report = build_report([(0, "a.py", 1, "TODO", "x"), (1, "b.py", 2, "TODO", "y")], vectors, {0: "alpha beta", 1: "beta gamma"}, {}, options)
    themes = report.themes
    assert isinstance(report, DebtReport)
    assert all(isinstance(theme, DebtTheme) for theme in themes)
    assert all(isinstance(match, DebtMatch) for theme in themes for match in theme.matches)
    assert themes[0].label == "beta / alpha"  # beta=2 first, then token-asc tie-break


def test_build_report_empty_corpus_is_clean() -> None:
    report = build_report([], np.zeros((0, 2), dtype=np.float32), {}, {}, DebtOptions())
    assert report == DebtReport(
        themes=(),
        scattered=0,
        markers_found=0,
        chunks_scanned=0,
        truncated=False,
        threshold=0.8,
        max_age_days=None,
        passed=None,
    )


@pytest.mark.parametrize("threshold", [0.5, 0.99])
def test_build_report_threshold_passthrough(threshold: float) -> None:
    report = build_report([], np.zeros((0, 2), dtype=np.float32), {}, {}, DebtOptions(threshold=threshold))
    assert report.threshold == threshold
