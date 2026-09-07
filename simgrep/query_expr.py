"""Boolean semantic query expressions: tokenizer, recursive-descent parser, evaluator.

Pure stdlib; no numpy. Grammar (Lucene-style, operators are UPPERCASE keywords only):

    expr    := or_expr
    or_expr := and_expr (OR and_expr)*
    and_expr:= unary ((AND)? unary)*      # adjacent atoms = implicit AND
    unary   := NOT unary | atom
    atom    := '(' expr ')' | WORD | QUOTED
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from simgrep.errors import ExprError

MAX_DEPTH = 32
MAX_NODES = 256

_OPERATOR_KINDS = {"AND": "and", "OR": "or", "NOT": "not"}


@dataclass(frozen=True)
class Leaf:
    text: str


@dataclass(frozen=True)
class Not:
    child: Expr


@dataclass(frozen=True)
class And:
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Or:
    left: Expr
    right: Expr


Expr = Leaf | Not | And | Or


@dataclass(frozen=True)
class Token:
    """kind in: lparen rparen and or not word quoted."""

    kind: str
    value: str
    pos: int


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            tokens.append(Token("lparen", ch, i))
            i += 1
        elif ch == ")":
            tokens.append(Token("rparen", ch, i))
            i += 1
        elif ch == '"':
            close = text.find('"', i + 1)
            if close == -1:
                raise ExprError(f"unterminated quoted phrase at position {i}", hint="close the quote")
            value = text[i + 1 : close]
            if not value:
                raise ExprError(f"empty quoted phrase at position {i}")
            tokens.append(Token("quoted", value, i))
            i = close + 1
        else:
            start = i
            while i < n and not text[i].isspace() and text[i] not in '()"':
                i += 1
            word = text[start:i]
            tokens.append(Token(_OPERATOR_KINDS.get(word, "word"), word, start))
    return tokens


def parse(text: str) -> Expr:
    parser = _Parser(tokenize(text), len(text))
    return parser.parse()


class _Parser:
    def __init__(self, tokens: list[Token], text_len: int) -> None:
        self._tokens = tokens
        self._i = 0
        self._text_len = text_len
        self._depth = 0
        self._nodes = 0
        self._last: Token | None = None

    def _register(self, token: Token) -> None:
        if self._nodes >= MAX_NODES:
            raise ExprError(
                f"expression exceeds maximum complexity {MAX_NODES} (position {token.pos})",
                hint="reduce the number of terms",
            )
        self._nodes += 1

    def _peek(self) -> Token | None:
        return self._tokens[self._i] if self._i < len(self._tokens) else None

    def _advance(self) -> Token:
        token = self._tokens[self._i]
        self._i += 1
        self._last = token
        return token

    def parse(self) -> Expr:
        if not self._tokens:
            raise ExprError("empty expression at position 0")
        node = self._parse_or()
        trailing = self._peek()
        if trailing is not None:
            raise ExprError(f"unexpected token {trailing.value!r} at position {trailing.pos}" " (unbalanced parenthesis or trailing input)")
        return node

    def _parse_or(self) -> Expr:
        node = self._parse_and()
        while (token := self._peek()) is not None and token.kind == "or":
            self._advance()
            self._register(token)
            node = Or(node, self._parse_and())
        return node

    def _parse_and(self) -> Expr:
        node = self._parse_unary()
        while (token := self._peek()) is not None:
            if token.kind == "and":
                self._advance()
                self._register(token)
                node = And(node, self._parse_unary())
            elif token.kind in ("word", "quoted", "lparen", "not"):
                # adjacent atoms = implicit AND
                self._register(token)
                node = And(node, self._parse_unary())
            else:
                break
        return node

    def _parse_unary(self) -> Expr:
        token = self._peek()
        if token is not None and token.kind == "not":
            self._advance()
            self._register(token)
            self._depth += 1
            try:
                if self._depth > MAX_DEPTH:
                    raise ExprError(
                        f"expression nesting exceeds maximum depth {MAX_DEPTH}" f" (position {token.pos})",
                        hint="simplify the expression",
                    )
                return Not(self._parse_unary())
            finally:
                self._depth -= 1
        return self._parse_atom()

    def _parse_atom(self) -> Expr:
        token = self._peek()
        if token is None:
            last = self._last
            if last is not None and last.kind in ("and", "or", "not"):
                raise ExprError(f"operator {last.value!r} at position {last.pos} is missing an operand")
            raise ExprError(f"expression ends where an operand was expected (position {self._text_len})")
        if token.kind == "lparen":
            self._advance()
            self._depth += 1
            if self._depth > MAX_DEPTH:
                raise ExprError(
                    f"expression nesting exceeds maximum depth {MAX_DEPTH}" f" (position {token.pos})",
                    hint="simplify the expression",
                )
            node = self._parse_or()
            closing = self._peek()
            if closing is None:
                raise ExprError(f"unbalanced parentheses: expected ')' matching '(' at position {token.pos}")
            if closing.kind != "rparen":
                raise ExprError(f"unexpected token {closing.value!r} at position {closing.pos}," f" expected ')' for '(' at position {token.pos}")
            self._advance()
            self._depth -= 1
            return node
        if token.kind in ("word", "quoted"):
            self._advance()
            self._register(token)
            return Leaf(token.value)
        raise ExprError(f"operator {token.value!r} at position {token.pos} is missing an operand")


def collect_leaves(expr: Expr) -> list[str]:
    seen: set[str] = set()
    leaves: list[str] = []

    def walk(node: Expr) -> None:
        match node:
            case Leaf():
                if node.text not in seen:
                    seen.add(node.text)
                    leaves.append(node.text)
            case Not():
                walk(node.child)
            case And() | Or():
                walk(node.left)
                walk(node.right)

    walk(expr)
    return leaves


def positive_leaves(expr: Expr) -> list[Leaf]:
    """All Leaf nodes with no Not ancestor, first-occurrence order.

    A leaf under any Not (at any depth) is negated and excluded; these are
    the leaves that define the per-label dominance threshold for NOT.
    """
    positive: list[Leaf] = []
    seen: set[str] = set()
    stack: list[tuple[Expr, bool]] = [(expr, False)]
    while stack:
        node, negated = stack.pop()
        match node:
            case Leaf():
                if not negated and node.text not in seen:
                    seen.add(node.text)
                    positive.append(node)
            case Not():
                stack.append((node.child, True))
            case And() | Or():
                # push right first so left pops first: first-occurrence order
                stack.append((node.right, negated))
                stack.append((node.left, negated))
    return positive


def evaluate(expr: Expr, leaf_scores: Mapping[str, Mapping[int, float]]) -> dict[int, float]:
    """Score the expression over the label universe of ALL leaf dicts.

    U = union of every leaf dict's keys. Each Leaf renders as a total over U
    (absent label = 0.0); And = elementwise min; Or = elementwise max — all
    over the same fixed universe.

    NOT is DOMINANCE-based: ``Not c`` renders 0.0 at a label where s_c(label)
    strictly exceeds the max score of every POSITIVE leaf at that label
    (the negated concept dominates all positive concepts), else 1.0.
    Positive leaves are the expression's Leaves with no Not ancestor (see
    :func:`positive_leaves`). Pure-NOT expressions (no positive leaves) fall
    back to fuzzy ``1 - s_c``, so ``NOT x`` and ``NOT NOT x`` behave as
    complement / identity. Rationale (real-model smoke 2026-08-24): fuzzy
    ``1 - s`` punished moderately-related docs harder than irrelevant ones,
    inverting rankings; dominance exclusion matches Lucene MUST_NOT intuition.

    Empty/absent leaf dicts give empty/zero results, never exceptions.
    """
    universe: set[int] = set()
    for scores in leaf_scores.values():
        universe.update(scores)
    labels = sorted(universe)

    def leaf_score(text: str, label: int) -> float:
        scores = leaf_scores.get(text)
        return float(scores.get(label, 0.0)) if scores is not None else 0.0

    positives = positive_leaves(expr)
    dominance = bool(positives)
    threshold: dict[int, float] = {label: max(leaf_score(p.text, label) for p in positives) for label in labels} if dominance else {}

    def render(node: Expr) -> dict[int, float]:
        match node:
            case Leaf():
                return {label: leaf_score(node.text, label) for label in labels}
            case Not():
                child = render(node.child)
                if not dominance:
                    # pure-NOT expression: fuzzy fallback
                    return {label: 1.0 - score for label, score in child.items()}
                return {label: 0.0 if score > threshold[label] else 1.0 for label, score in child.items()}
            case And():
                left = render(node.left)
                right = render(node.right)
                return {label: min(left[label], right[label]) for label in labels}
            case Or():
                left = render(node.left)
                right = render(node.right)
                return {label: max(left[label], right[label]) for label in labels}
        raise ExprError(f"unknown expression node: {node!r}")  # pragma: no cover

    return render(expr)
