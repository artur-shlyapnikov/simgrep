"""Contract: --lexical-fallback off/fill/empty semantics pinned at the ranking layer.

The e2e surface cannot produce lexical-only rows (effective_candidate_top >= top keeps
every output slot semantic under the fake ANN), so the numeric mode behavior is pinned here,
directly against rank_candidates."""

from pathlib import Path
from typing import Any

import pytest

from simgrep.corpus import StoredChunk
from simgrep.models import LexicalFallbackMode, SearchOptions
from simgrep.ranking import rank_candidates
from tests.conftest import _ranking_chunk


def _options(mode: LexicalFallbackMode) -> SearchOptions:
    return SearchOptions(query="needle", top=10, lexical_weight=0.5, lexical_fallback=mode)


def _rank(mode: LexicalFallbackMode) -> list[dict[str, Any]]:
    semantic_rows = [
        _ranking_chunk(1, Path("a.py"), "alpha", "source"),
        _ranking_chunk(2, Path("b.py"), "beta", "source"),
    ]
    lexical_rows: list[tuple[StoredChunk, float]] = [(_ranking_chunk(3, Path("c.md"), "needle needle needle", "docs"), 2.0)]
    ranked = rank_candidates(
        query="needle",
        semantic_matches=[(1, 0.9), (2, 0.8)],
        semantic_rows=semantic_rows,
        lexical_rows=lexical_rows,
        options=_options(mode),
    )
    return [{"path": r.file_path.name, "score": r.score, "lexonly": bool(r.why.get("lexical_only"))} for r in ranked]


@pytest.mark.parametrize("mode", list(LexicalFallbackMode), ids=lambda m: m.value)
def test_fallback_mode_contract(mode: LexicalFallbackMode) -> None:
    rows = _rank(mode)
    lex_only = [row for row in rows if row["lexonly"]]
    semantic = [row for row in rows if not row["lexonly"]]
    assert semantic, "semantic rows must always surface"
    if mode == LexicalFallbackMode.off:
        assert lex_only == []
    elif mode == LexicalFallbackMode.fill:
        # cap = min(0.35, max(min semantic score - 0.001, 0)); assert the two observable halves
        assert all(row["score"] <= min(r["score"] for r in semantic) for row in lex_only)
        assert all(row["score"] <= 0.35 for row in lex_only)
    else:
        assert all(row["score"] == 0.0 for row in lex_only)


@pytest.mark.parametrize("mode", [LexicalFallbackMode.fill, LexicalFallbackMode.empty], ids=lambda m: m.value)
def test_no_semantic_items_keeps_raw_lexical_scores(mode: LexicalFallbackMode) -> None:
    ranked = rank_candidates(
        query="needle",
        semantic_matches=[],
        semantic_rows=[],
        lexical_rows=[(_ranking_chunk(3, Path("c.md"), "needle", "docs"), 2.0)],
        options=_options(mode),
    )
    assert [r.file_path.name for r in ranked] == ["c.md"]
    assert all(r.score > 0.0 for r in ranked)


def test_off_drops_everything_without_semantic_items() -> None:
    ranked = rank_candidates(
        query="needle",
        semantic_matches=[],
        semantic_rows=[],
        lexical_rows=[(_ranking_chunk(3, Path("c.md"), "needle", "docs"), 2.0)],
        options=_options(LexicalFallbackMode.off),
    )
    assert ranked == []
