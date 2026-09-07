"""Budgeted context assembly: greedy MMR selection of candidates under a token budget.

Pure domain core for `simgrep pack` — stdlib only, no I/O. See docs: search finds,
pack assembles; one call returns a deduplicated, budget-fitting selection with
per-pick gains and dropped accounting.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace

_ALPHA_TOKEN_RE = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class PackCandidate:
    """One retrievable chunk offered to the packer."""

    label: int
    path: str
    line_start: int
    line_end: int
    score: float
    text: str


@dataclass(frozen=True)
class PackSelection:
    """A candidate chosen by the greedy loop, in pick order."""

    candidate: PackCandidate
    tokens: int
    truncated: bool
    gain: float


@dataclass(frozen=True)
class PackOutcome:
    """Result of packing: selections plus budget accounting."""

    selections: list[PackSelection]
    used_tokens: int
    pool_size: int
    dropped: int
    budget: int


def estimate_tokens(text: str) -> int:
    """Documented chars/4 approximation, never below 1."""
    return max(1, (len(text) + 3) // 4)


def jaccard(a: str, b: str) -> float:
    """Jaccard similarity over lowercase alpha-token sets; either side empty -> 0.0."""
    tokens_a = set(_ALPHA_TOKEN_RE.findall(a.lower()))
    tokens_b = set(_ALPHA_TOKEN_RE.findall(b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _pick_index(candidates: list[PackCandidate], gains: list[float]) -> int:
    """Index of the best candidate under the total tie-break chain."""
    return min(
        range(len(candidates)),
        key=lambda i: (
            -gains[i],
            -candidates[i].score,
            candidates[i].path,
            candidates[i].line_start,
            candidates[i].label,
        ),
    )


def pack_candidates(
    candidates: list[PackCandidate],
    budget: int,
    *,
    lam: float = 0.7,
    estimator: Callable[[str], int] = estimate_tokens,
    redundancy: Callable[[str, str], float] = jaccard,
) -> PackOutcome:
    """Greedily select candidates under ``budget`` tokens via MMR-style gains.

    Each round picks the argmax of ``lam*score - (1-lam)*max_redundancy``
    (redundancy term is 0 before any pick), using the deterministic tie-break
    chain ``(-gain, -score, path, line_start, label)``. Candidates that no
    longer fit are dropped permanently. If nothing ever fits, the single
    best-gain candidate is truncated to the budget instead.
    """
    if budget <= 0:
        raise ValueError(f"budget must be a positive token count, got {budget}")

    pool_size = len(candidates)
    remaining = list(candidates)
    selected_texts: list[str] = []
    selections: list[PackSelection] = []
    used = 0

    while remaining:
        gains = [lam * cand.score - (1 - lam) * max((redundancy(cand.text, t) for t in selected_texts), default=0.0) for cand in remaining]
        best_i = _pick_index(remaining, gains)
        best_gain = gains[best_i]
        best = remaining.pop(best_i)
        tokens = estimator(best.text)
        if used + tokens <= budget:
            selections.append(PackSelection(best, tokens, False, best_gain))
            selected_texts.append(best.text)
            used += tokens

    if not selections and pool_size:
        # Nothing fit: fall back to the single best-gain candidate, truncated.
        gains = [cand.score * lam for cand in candidates]
        best = candidates[_pick_index(candidates, gains)]
        truncated_text = best.text[:budget] + "…"
        selections.append(PackSelection(replace(best, text=truncated_text), budget, True, lam * best.score))
        used = budget

    return PackOutcome(
        selections=selections,
        used_tokens=used,
        pool_size=pool_size,
        dropped=pool_size - len(selections),
        budget=budget,
    )
