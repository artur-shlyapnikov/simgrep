"""Pure-math tests for simgrep.pack: token estimation, jaccard, greedy MMR budget fit."""

from __future__ import annotations

import pytest

from simgrep.pack import (
    PackCandidate,
    estimate_tokens,
    jaccard,
    pack_candidates,
)


def _cand(
    label: int,
    score: float,
    text: str,
    *,
    path: str = "src/a.py",
    line_start: int = 1,
) -> PackCandidate:
    return PackCandidate(
        label=label,
        path=path,
        line_start=line_start,
        line_end=line_start + 10,
        score=score,
        text=text,
    )


# ---------------------------------------------------------------- token estimator


def test_estimate_tokens_chars_over_four() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("ab") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("x" * 8) == 2
    assert estimate_tokens("x" * 9) == 3


def test_estimate_tokens_never_below_one() -> None:
    for size in range(0, 6):
        assert estimate_tokens("x" * size) >= 1


# ---------------------------------------------------------------- jaccard


def test_jaccard_identical_is_one() -> None:
    assert jaccard("def foo():\n    return 1", "DEF FOO(): RETURN 1") == 1.0


def test_jaccard_disjoint_is_zero() -> None:
    assert jaccard("alpha beta", "gamma delta") == 0.0


def test_jaccard_partial_overlap() -> None:
    # {alpha, beta} vs {beta, gamma} -> 1/3
    assert jaccard("alpha beta", "beta gamma") == pytest.approx(1 / 3)


def test_jaccard_empty_side_is_zero() -> None:
    assert jaccard("", "alpha beta") == 0.0
    assert jaccard("alpha beta", "") == 0.0
    assert jaccard("123 456!", "alpha") == 0.0  # no alpha tokens


def test_jaccard_case_and_punctuation_insensitive() -> None:
    assert jaccard("Alpha, BETA!", "alpha beta") == 1.0


# ---------------------------------------------------------------- budget validation


@pytest.mark.parametrize("bad_budget", [0, -1, -100])
def test_non_positive_budget_raises(bad_budget: int) -> None:
    pool = [_cand(1, 0.9, "some text")]
    with pytest.raises(ValueError):
        pack_candidates(pool, bad_budget)


# ---------------------------------------------------------------- empty pool


def test_empty_pool_yields_zero_selections() -> None:
    outcome = pack_candidates([], 100)
    assert outcome.selections == []
    assert outcome.used_tokens == 0
    assert outcome.pool_size == 0
    assert outcome.dropped == 0
    assert outcome.budget == 100


# ---------------------------------------------------------------- greedy fit


def test_everything_fits_selects_all_in_gain_order() -> None:
    pool = [
        _cand(1, 0.5, "w" * 20),
        _cand(2, 0.9, "distinct one"),
        _cand(3, 0.7, "distinct two"),
    ]
    outcome = pack_candidates(pool, 1000, lam=1.0)
    assert [s.candidate.label for s in outcome.selections] == [2, 3, 1]
    assert all(not s.truncated for s in outcome.selections)
    assert outcome.dropped == 0
    assert outcome.used_tokens == sum(s.tokens for s in outcome.selections)


def test_exact_fit_boundary_selects() -> None:
    # First pick uses 10 tokens (40 chars), leaving exactly 10 for the next.
    pool = [
        _cand(1, 0.9, "x" * 40),
        _cand(2, 0.5, "y" * 40),
    ]
    outcome = pack_candidates(pool, 20)
    assert [s.candidate.label for s in outcome.selections] == [1, 2]
    assert outcome.used_tokens == 20
    assert outcome.dropped == 0


def test_one_token_over_boundary_drops() -> None:
    pool = [
        _cand(1, 0.9, "x" * 40),  # 10 tokens
        _cand(2, 0.5, "y" * 41),  # 11 tokens > 10 remaining
    ]
    outcome = pack_candidates(pool, 20)
    assert [s.candidate.label for s in outcome.selections] == [1]
    assert outcome.dropped == 1
    assert outcome.pool_size == 2


def test_feasibility_skip_drops_oversized_and_continues() -> None:
    pool = [
        _cand(1, 0.95, "z" * 400),  # 100 tokens, never fits budget 50
        _cand(2, 0.80, "fits nicely"),
        _cand(3, 0.70, "also fits"),
    ]
    outcome = pack_candidates(pool, 50, lam=1.0)
    # Candidate 1 is picked first (highest gain), found oversized, dropped;
    # the loop continues with the next-best candidates.
    assert [s.candidate.label for s in outcome.selections] == [2, 3]
    assert not any(s.truncated for s in outcome.selections)
    assert outcome.pool_size == 3
    assert outcome.dropped == 1
    assert outcome.used_tokens == sum(s.tokens for s in outcome.selections)


# ---------------------------------------------------------------- MMR diversity


def test_mmr_demotes_near_duplicate_when_lam_below_one() -> None:
    dup_a = " ".join(["token"] * 30)
    dup_b = " ".join(["token"] * 29) + " extra"
    distinct = "completely different words here"
    pool = [
        _cand(1, 0.90, dup_a),
        _cand(2, 0.85, dup_b),  # near-duplicate of candidate 1
        _cand(3, 0.80, distinct),
    ]

    pure_relevance = pack_candidates(pool, 1000, lam=1.0)
    assert [s.candidate.label for s in pure_relevance.selections] == [1, 2, 3]

    diversified = pack_candidates(pool, 1000, lam=0.5)
    labels = [s.candidate.label for s in diversified.selections]
    # Near-duplicate is demoted behind the distinct candidate.
    assert labels == [1, 3, 2]
    # Its gain reflects the redundancy penalty.
    dup_sel = next(s for s in diversified.selections if s.candidate.label == 2)
    assert dup_sel.gain < 0.5 * 0.85


def test_mmr_no_penalty_before_first_selection() -> None:
    pool = [_cand(1, 0.9, "only one")]
    outcome = pack_candidates(pool, 1000, lam=0.5)
    assert outcome.selections[0].gain == pytest.approx(0.45)


# ---------------------------------------------------------------- truncation fallback


def test_oversized_fallback_truncates_single_best() -> None:
    pool = [
        _cand(1, 0.70, "b" * 400),
        _cand(2, 0.90, "a" * 400),
        _cand(3, 0.80, "c" * 400),
    ]
    outcome = pack_candidates(pool, 50)
    assert len(outcome.selections) == 1
    sel = outcome.selections[0]
    assert sel.candidate.label == 2  # best gain wins the fallback
    assert sel.truncated is True
    assert sel.tokens == 50
    assert sel.gain == pytest.approx(0.63)
    assert len(sel.candidate.text) == 51
    assert sel.candidate.text.endswith("…")
    assert sel.candidate.text.startswith("a")
    assert outcome.used_tokens == 50
    assert outcome.dropped == 2  # pool_size - selections, truncation pick excluded
    assert outcome.used_tokens <= outcome.budget


def test_no_fallback_when_something_fit() -> None:
    # One small candidate is selected; the rest are oversized and merely dropped.
    pool = [
        _cand(1, 0.60, "small"),
        _cand(2, 0.95, "h" * 400),
    ]
    outcome = pack_candidates(pool, 10)
    assert [s.candidate.label for s in outcome.selections] == [1]
    assert all(not s.truncated for s in outcome.selections)
    assert outcome.dropped == 1


# ---------------------------------------------------------------- determinism & tie-breaks


def test_same_input_gives_identical_outcome() -> None:
    pool = [_cand(1, 0.9, f"text {i} shared words everywhere", path=f"src/{i}.py") for i in range(6)]
    first = pack_candidates([*pool], 60)
    second = pack_candidates([*pool], 60)
    assert first == second


def test_tie_break_chain_path_then_line_then_label() -> None:
    text = "identical content"
    pool = [
        _cand(9, 0.5, text, path="src/b.py"),
        _cand(4, 0.5, text, path="src/a.py", line_start=20),
        _cand(2, 0.5, text, path="src/a.py", line_start=5),
        _cand(7, 0.5, text, path="src/a.py", line_start=5),
    ]
    outcome = pack_candidates(pool, 1000, lam=1.0)
    order = [(s.candidate.path, s.candidate.line_start, s.candidate.label) for s in outcome.selections]
    assert order == [
        ("src/a.py", 5, 2),
        ("src/a.py", 5, 7),
        ("src/a.py", 20, 4),
        ("src/b.py", 1, 9),
    ]


# ---------------------------------------------------------------- accounting


def test_used_tokens_equals_sum_of_selection_tokens() -> None:
    pool = [_cand(i, 1.0 - i * 0.05, f"c{i} " * (i + 1)) for i in range(5)]
    outcome = pack_candidates(pool, 40)
    assert outcome.used_tokens == sum(s.tokens for s in outcome.selections)
    assert outcome.used_tokens <= outcome.budget
    assert outcome.dropped == outcome.pool_size - len(outcome.selections)


def test_custom_estimator_and_redundancy_are_honored() -> None:
    pool = [
        _cand(1, 0.9, "aaaa"),
        _cand(2, 0.8, "bbbb"),
    ]
    outcome = pack_candidates(
        pool,
        3,
        lam=0.5,
        estimator=len,
        redundancy=lambda a, b: 1.0,
    )
    # len("aaaa") == 4 > 3 -> oversized; len("bbbb") == 4 > 3 too.
    assert len(outcome.selections) == 1
    assert outcome.selections[0].truncated is True
    assert outcome.selections[0].tokens == 3
