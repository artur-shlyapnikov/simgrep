"""Property-based tests for the `query_expr` evaluator/parser algebra (Round 20).

`evaluate` is a pure function over score dicts, so commutativity,
associativity, idempotence, totality/bounds and dominance-boundary properties
are checked over arbitrary expression trees built with recursive hypothesis
strategies. Parser robustness is fuzzed directly (P7/P8); the two reachable
uncovered error lines are pinned with exact messages (P9).
"""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from simgrep.errors import ExprError
from simgrep.query_expr import (
    MAX_DEPTH,
    And,
    Expr,
    Leaf,
    Not,
    Or,
    collect_leaves,
    evaluate,
    parse,
    positive_leaves,
)

LABELS = range(0, 6)
LEAF_TEXTS = ("a", "b", "c")
SCORE_MAPS = st.fixed_dictionaries({text: st.dictionaries(st.integers(0, 5), st.floats(min_value=0.0, max_value=1.0), max_size=4) for text in LEAF_TEXTS})
ATOMS = st.sampled_from([Leaf(text) for text in LEAF_TEXTS])


def _combine(inner: st.SearchStrategy[Expr]) -> st.SearchStrategy[Expr]:
    return st.builds(And, inner, inner) | st.builds(Or, inner, inner) | st.builds(Not, inner)


EXPR_TREE: st.SearchStrategy[Expr] = st.recursive(ATOMS, _combine, max_leaves=6)
PURE_NOT_TREE: st.SearchStrategy[Expr] = st.recursive(st.builds(Not, ATOMS), _combine, max_leaves=5)

# --- P1 totality + bounds -------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(EXPR_TREE, SCORE_MAPS)
def test_evaluate_output_is_total_over_universe_and_bounded(expr: Expr, scores: dict[str, dict[int, float]]) -> None:
    """evaluate is defined exactly over the union of all leaf-dict labels, in [0.0, 1.0]."""
    universe: set[int] = set()
    for mapping in scores.values():
        universe.update(mapping)
    out = evaluate(expr, scores)
    assert set(out) == universe
    assert all(0.0 <= value <= 1.0 for value in out.values())


# --- P2 commutativity -------------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(EXPR_TREE, EXPR_TREE, SCORE_MAPS)
def test_and_or_fusion_is_commutative(left: Expr, right: Expr, scores: dict[str, dict[int, float]]) -> None:
    """And/Or fusion ignores operand order, even under dominance-NOT subtrees."""
    assert evaluate(And(left, right), scores) == evaluate(And(right, left), scores)
    assert evaluate(Or(left, right), scores) == evaluate(Or(right, left), scores)


# --- P3 associativity -------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(EXPR_TREE, EXPR_TREE, EXPR_TREE, SCORE_MAPS)
def test_and_or_fusion_is_associative(left: Expr, middle: Expr, right: Expr, scores: dict[str, dict[int, float]]) -> None:
    """min/max chains associate regardless of nesting shape; threshold computation is order-independent."""
    assert evaluate(And(And(left, middle), right), scores) == evaluate(And(left, And(middle, right)), scores)
    assert evaluate(Or(Or(left, middle), right), scores) == evaluate(Or(left, Or(middle, right)), scores)


# --- P4 idempotence ---------------------------------------------------------------------


@settings(max_examples=40, deadline=None)
@given(EXPR_TREE, SCORE_MAPS)
def test_and_or_are_idempotent(expr: Expr, scores: dict[str, dict[int, float]]) -> None:
    """Re-conjuncting / re-disjuncting an identical subtree never changes scores."""
    assert evaluate(And(expr, expr), scores) == evaluate(expr, scores)
    assert evaluate(Or(expr, expr), scores) == evaluate(expr, scores)


# --- P5 pure-NOT double-negation identity ------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(PURE_NOT_TREE, SCORE_MAPS)
def test_pure_not_double_negation_identity(expr: Expr, scores: dict[str, dict[int, float]]) -> None:
    """For any expression with NO positive leaves, NOT NOT e == e up to IEEE double
    rounding (each NOT applies 1-s; two invocations may differ by 1 ulp,
    e.g. s=0.1 -> 0.09999999999999998). Approx equality pins the composition
    contract, not bit-exactness."""
    assert positive_leaves(expr) == []
    assert evaluate(Not(Not(expr)), scores) == pytest.approx(evaluate(expr, scores))


# --- P6 dominance boundary is STRICT ------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
    st.integers(min_value=0, max_value=5),
)
def test_dominance_boundary_is_strict(positive: float, negated: float, label: int) -> None:
    """p AND NOT n renders exactly p unless n strictly exceeds every positive score; n == p keeps the doc."""
    scores = {"p": {label: positive}, "n": {label: negated}}
    out = evaluate(parse("p AND NOT n"), scores)
    assert out[label] == (0.0 if negated > positive else positive)


# --- P7 parser robustness fuzz ------------------------------------------------------------

FUZZ_ALPHABET = 'ab() "ANDORNOT\t\n\u00e9\u6f22'


@settings(max_examples=300, deadline=None)
@given(st.text(alphabet=FUZZ_ALPHABET, min_size=1, max_size=24).filter(lambda s: s.strip() != ""))
def test_parse_never_raises_non_expr_error_and_positions_in_range(text: str) -> None:
    """On arbitrary input, parse either returns a walkable Expr or raises ExprError;
    every integer in an error message is a position <= len(text) or the depth cap."""
    try:
        expr = parse(text)
    except ExprError as error:
        for number in re.findall(r"\d+", str(error)):
            assert int(number) <= len(text) or int(number) == MAX_DEPTH
    else:
        assert all(isinstance(leaf, str) for leaf in collect_leaves(expr))


# --- P8 whitespace insensitivity -----------------------------------------------------------

TOKENS = st.lists(
    st.sampled_from(["a", "b", "c", "AND", "OR", "NOT", "(a)", '"p q"', "(a OR b)", "NOT a"]),
    min_size=1,
    max_size=8,
)


@settings(max_examples=250, deadline=None)
@given(TOKENS)
def test_ast_is_whitespace_insensitive(tokens: list[str]) -> None:
    """Any spacing between tokens (including tabs/newlines/runs) yields the identical AST."""
    try:
        base = parse(" ".join(tokens))
    except ExprError:
        return  # unparseable sequences carry no AST equality claim
    separators = (" ", "  ", "\t", "\n")
    joined = "".join(token + (separators[(index + len(tokens)) % 4] if index < len(tokens) - 1 else "") for index, token in enumerate(tokens))
    assert parse(joined) == base


# --- P9 uncovered error lines 76 + 155 -------------------------------------------------------


def test_uncovered_error_lines_fire_with_exact_positions() -> None:
    """Line 76 (unterminated quote) and line 155 (EOS where operand expected) report
    the byte position of the offense; line-155's position equals len(text)."""
    cases = [
        ('a "unclosed', "unterminated quoted phrase at position 2"),
        ('"x', "unterminated quoted phrase at position 0"),
        ("(", "expression ends where an operand was expected (position 1)"),
        ("a OR (", "expression ends where an operand was expected (position 6)"),
    ]
    for text, expected in cases:
        with pytest.raises(ExprError) as excinfo:
            parse(text)
        assert str(excinfo.value) == expected
        if "operand was expected" in expected:
            assert int(expected.rsplit(" ", 1)[1][:-1]) == len(text)
