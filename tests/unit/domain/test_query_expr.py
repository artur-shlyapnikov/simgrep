"""Tests for simgrep.query_expr — tokenizer, parser, collector, evaluator."""

import pytest

from simgrep.errors import ExprError
from simgrep.query_expr import (
    And,
    Leaf,
    Not,
    Or,
    Token,
    collect_leaves,
    evaluate,
    parse,
    positive_leaves,
    tokenize,
)


class TestTokenize:
    def test_kinds_and_positions(self) -> None:
        tokens = tokenize('(auth OR login) AND NOT "connection pool"')
        assert tokens == [
            Token(kind="lparen", value="(", pos=0),
            Token(kind="word", value="auth", pos=1),
            Token(kind="or", value="OR", pos=6),
            Token(kind="word", value="login", pos=9),
            Token(kind="rparen", value=")", pos=14),
            Token(kind="and", value="AND", pos=16),
            Token(kind="not", value="NOT", pos=20),
            Token(kind="quoted", value="connection pool", pos=24),
        ]

    def test_lowercase_operators_are_words(self) -> None:
        kinds = [t.kind for t in tokenize("and or not And")]
        assert kinds == ["word", "word", "word", "word"]

    def test_quoted_phrase_keeps_spaces(self) -> None:
        (token,) = tokenize('"error   code"')
        assert token.kind == "quoted"
        assert token.value == "error   code"


class TestParse:
    def test_precedence_not_over_and_over_or(self) -> None:
        # a OR b AND c  ==> a OR (b AND c)
        expr = parse("a OR b AND c")
        assert expr == Or(Leaf("a"), And(Leaf("b"), Leaf("c")))
        # NOT a AND b  ==> (NOT a) AND b
        expr2 = parse("NOT a AND b")
        assert expr2 == And(Not(Leaf("a")), Leaf("b"))
        # NOT binds tighter than AND
        assert parse("NOT NOT a") == Not(Not(Leaf("a")))

    def test_parens_override_precedence(self) -> None:
        expr = parse("(a OR b) AND c")
        assert expr == And(Or(Leaf("a"), Leaf("b")), Leaf("c"))

    def test_implicit_and_between_adjacent_atoms(self) -> None:
        assert parse("auth login") == And(Leaf("auth"), Leaf("login"))
        assert parse('auth "connection pool" (retry OR backoff)') == And(
            And(Leaf("auth"), Leaf("connection pool")),
            Or(Leaf("retry"), Leaf("backoff")),
        )
        # implicit AND before NOT
        assert parse("auth NOT oauth") == And(Leaf("auth"), Not(Leaf("oauth")))

    def test_single_word_and_single_quoted(self) -> None:
        assert parse("hello") == Leaf("hello")
        assert parse('"two words"') == Leaf("two words")

    def test_empty_expression_raises_with_pos(self) -> None:
        for text in ("", "   "):
            with pytest.raises(ExprError) as excinfo:
                parse(text)
            assert "0" in str(excinfo.value)

    def test_unbalanced_opening_paren(self) -> None:
        with pytest.raises(ExprError) as excinfo:
            parse("(a OR b")
        assert "position" in str(excinfo.value)
        assert excinfo.value.hint is None or isinstance(excinfo.value.hint, str)

    def test_unbalanced_closing_paren_is_trailing_token(self) -> None:
        with pytest.raises(ExprError):
            parse("a) OR b")

    def test_trailing_tokens_after_top_level_expr(self) -> None:
        with pytest.raises(ExprError) as excinfo:
            parse("auth) AND x")
        assert "4" in str(excinfo.value)  # 0-based pos of ')'

    def test_operator_missing_operand(self) -> None:
        with pytest.raises(ExprError) as open_err:
            parse("a AND")
        assert "position" in str(open_err.value)
        with pytest.raises(ExprError) as lead_err:
            parse("OR b")
        assert "0" in str(lead_err.value)
        with pytest.raises(ExprError):
            parse("NOT")

    def test_empty_quoted_leaf_raises(self) -> None:
        with pytest.raises(ExprError) as excinfo:
            parse('a ""')
        assert "2" in str(excinfo.value)

    def test_depth_cap_32(self) -> None:
        ok = "(" * 32 + "x" + ")" * 32
        assert parse(ok) == Leaf("x")
        too_deep = "(" * 33 + "x" + ")" * 33
        with pytest.raises(ExprError) as excinfo:
            parse(too_deep)
        assert excinfo.value.hint == "simplify the expression"

    def test_not_chain_depth_cap_32(self) -> None:
        expr = parse("NOT " * 32 + "x")
        node: Leaf | Not = Leaf("x")
        for _ in range(32):
            node = Not(node)
        assert expr == node

        with pytest.raises(ExprError) as excinfo:
            parse("NOT " * 33 + "x")
        assert excinfo.value.hint == "simplify the expression"

    def test_mixed_paren_not_depth_cap(self) -> None:
        with pytest.raises(ExprError) as excinfo:
            parse("(" * 20 + "NOT " * 20 + "x" + ")" * 20)
        assert excinfo.value.hint == "simplify the expression"

    def test_error_positions_are_zero_based(self) -> None:
        with pytest.raises(ExprError) as excinfo:
            parse("auth AND")
        # 'AND' starts at index 5; message must reference a 0-based position
        assert "5" in str(excinfo.value)


class TestNodeCap:
    def test_wide_or_chain_raises_complexity(self) -> None:
        with pytest.raises(ExprError) as excinfo:
            parse(" OR ".join(str(i) for i in range(200)))
        assert "maximum complexity" in str(excinfo.value)
        assert excinfo.value.hint == "reduce the number of terms"

    def test_wide_implicit_and_chain_raises_complexity(self) -> None:
        with pytest.raises(ExprError) as excinfo:
            parse(" ".join(str(i) for i in range(200)))
        assert "maximum complexity" in str(excinfo.value)
        assert excinfo.value.hint == "reduce the number of terms"

    def test_or_chain_below_cap_succeeds(self) -> None:
        expr = parse(" OR ".join(str(i) for i in range(100)))
        assert collect_leaves(expr) == [str(i) for i in range(100)]

    def test_repeated_leaf_chain_raises_complexity_not_recursion(self) -> None:
        with pytest.raises(ExprError) as excinfo:
            parse(" OR ".join(["a"] * 5000))
        assert "maximum complexity" in str(excinfo.value)

    def test_depth_cap_still_fires_before_node_cap(self) -> None:
        with pytest.raises(ExprError) as excinfo:
            parse("NOT " * 40 + "x")
        assert "maximum depth" in str(excinfo.value)
        assert excinfo.value.hint == "simplify the expression"


class TestCollectLeaves:
    def test_first_occurrence_order_deduped(self) -> None:
        expr = parse('b AND (a OR b) AND NOT "b a"')
        assert collect_leaves(expr) == ["b", "a", "b a"]

    def test_single_leaf(self) -> None:
        assert collect_leaves(parse("solo")) == ["solo"]


class TestEvaluate:
    LEAVES = {
        "auth": {1: 0.9, 2: 0.4},
        "oauth": {2: 0.8},
        "retry": {},
    }

    def test_leaf_lookup_missing_label_is_zero(self) -> None:
        result = evaluate(Leaf("auth"), self.LEAVES)
        assert result == {1: 0.9, 2: 0.4}
        # label absent from the leaf's dict renders as implicit 0.0 over the universe
        assert evaluate(And(Leaf("auth"), Leaf("oauth")), self.LEAVES) == {
            1: min(0.9, 0.0),
            2: min(0.4, 0.8),
        }

    def test_absent_leaf_renders_zeros_over_universe(self) -> None:
        assert evaluate(Leaf("missing"), self.LEAVES) == {1: 0.0, 2: 0.0}
        assert evaluate(Leaf("retry"), self.LEAVES) == {1: 0.0, 2: 0.0}

    def test_pure_not_falls_back_to_fuzzy(self) -> None:
        # no positive leaves ⇒ fuzzy fallback 1 - s over the universe
        result = evaluate(Not(Leaf("oauth")), self.LEAVES)
        assert result == {1: pytest.approx(1.0), 2: pytest.approx(0.2)}

    def test_and_min_over_universe(self) -> None:
        # elementwise min over the fixed universe; empty leaf dict contributes 0.0
        result = evaluate(And(Leaf("auth"), Leaf("retry")), self.LEAVES)
        assert result == {1: 0.0, 2: 0.0}

    def test_or_max_over_universe(self) -> None:
        result = evaluate(Or(Leaf("auth"), Leaf("oauth")), self.LEAVES)
        assert result == {1: 0.9, 2: max(0.4, 0.8)}

    def test_union_propagation_bottom_up(self) -> None:
        expr = Or(And(Leaf("auth"), Leaf("oauth")), Leaf("auth"))
        result = evaluate(expr, self.LEAVES)
        # And: {1: min(0.9, 0)=0.0, 2: min(0.4, 0.8)=0.4}; Or with auth lifts label 1
        assert result == {1: 0.9, 2: 0.4}

    def test_and_not_excludes_negated_and_keeps_absent_docs(self) -> None:
        leaves = {"retry": {1: 0.9}, "oauth": {2: 0.95}}
        expr = parse("retry AND NOT oauth")
        result = evaluate(expr, leaves)
        # dominance threshold = max positive leaf score per label ({1: 0.9, 2: 0.0});
        # oauth dominates at label 2 (excluded), is absent at label 1 (kept, NOT=1.0)
        assert result == {1: pytest.approx(0.9), 2: pytest.approx(0.0)}

    def test_dominance_exclusion_negated_beats_every_positive(self) -> None:
        # real-model smoke failure mode: auth doc moderately similar to oauth must be
        # excluded because oauth's score dominates every positive leaf at that label
        leaves = {"auth": {1: 0.5}, "oauth": {1: 0.9}}
        result = evaluate(parse("auth AND NOT oauth"), leaves)
        assert result == {1: 0.0}

    def test_mere_relatedness_is_kept(self) -> None:
        # negated concept scores below some positive leaf ⇒ doc kept (NOT renders 1.0)
        leaves = {"auth": {1: 0.9}, "cache": {1: 0.3}}
        result = evaluate(parse("auth AND NOT cache"), leaves)
        assert result == {1: pytest.approx(0.9)}

    def test_tie_with_positive_is_kept(self) -> None:
        # strict dominance: equal scores are NOT excluded
        leaves = {"auth": {1: 0.7}, "cache": {1: 0.7}}
        result = evaluate(parse("auth AND NOT cache"), leaves)
        assert result == {1: pytest.approx(0.7)}

    def test_double_negation_pure_fuzzy_equals_leaf(self) -> None:
        # NOT NOT oauth has no positive leaves ⇒ both NOTs fuzzy: 1 - (1 - s) = s
        result = evaluate(parse("NOT NOT oauth"), self.LEAVES)
        assert result == {1: pytest.approx(0.0), 2: pytest.approx(0.8)}

    def test_triple_negation_pure_fuzzy_equals_single_not(self) -> None:
        triple = evaluate(parse("NOT NOT NOT oauth"), self.LEAVES)
        single = evaluate(parse("NOT oauth"), self.LEAVES)
        assert triple == single

    def test_not_nested_under_or_uses_dominance(self) -> None:
        # positive leaves of the whole expression (just "a") set the threshold;
        # NOT b excludes only where b's score exceeds a's score
        leaves = {"a": {1: 0.9, 2: 0.4}, "b": {1: 0.3, 2: 0.95}}
        result = evaluate(parse("a OR NOT b"), leaves)
        # NOT b: label 1: 0.3 <= 0.9 ⇒ 1.0; label 2: 0.95 > 0.4 ⇒ 0.0
        assert result == {1: pytest.approx(1.0), 2: pytest.approx(0.4)}

    def test_combined_expression_matches_spec_example(self) -> None:
        leaves = {"auth": {7: 0.9}, "login": {8: 0.85}, "oauth": {7: 0.95}}
        expr = parse("(auth OR login) AND NOT oauth")
        result = evaluate(expr, leaves)
        # threshold = max(auth, login) per label = {7: 0.9, 8: 0.85};
        # oauth dominates at label 7 ⇒ excluded; absent at label 8 ⇒ kept
        assert result == {7: pytest.approx(0.0), 8: pytest.approx(0.85)}

    def test_positive_leaves_excludes_negated_subtrees(self) -> None:
        assert positive_leaves(parse("a AND NOT b")) == [Leaf("a")]
        # true first-occurrence (left-to-right) order across subtrees
        assert positive_leaves(parse("a OR (c AND d)")) == [Leaf("a"), Leaf("c"), Leaf("d")]
        assert positive_leaves(parse("NOT x OR y")) == [Leaf("y")]
        assert positive_leaves(parse("NOT (b OR NOT c)")) == []

    def test_empty_scores_deterministic(self) -> None:
        expr = parse("a AND (b OR NOT c)")
        first = evaluate(expr, {})
        second = evaluate(expr, {})
        assert first == second == {}
