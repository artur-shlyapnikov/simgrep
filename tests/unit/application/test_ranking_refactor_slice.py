from __future__ import annotations

from pathlib import Path

from simgrep.models import DiversityMode, FileRole, LexicalFallbackMode, PathBoost, SearchOptions
from simgrep.ranking import rank_candidates
from tests.conftest import _ranking_chunk


def _options(
    *,
    top: int = 5,
    lexical_top: int = 20,
    lexical_weight: float = 0.3,
    diversity: DiversityMode = DiversityMode.none,
    path_boosts: tuple[PathBoost, ...] = (),
    lexical_fallback: LexicalFallbackMode = LexicalFallbackMode.fill,
    min_score: float = 0.0,
) -> SearchOptions:
    return SearchOptions(
        query="rollback payment",
        top=top,
        min_score=min_score,
        lexical_top=lexical_top,
        lexical_weight=lexical_weight,
        diversity=diversity,
        path_boosts=path_boosts,
        lexical_fallback=lexical_fallback,
    )


class TestSemanticNormalization:
    def test_semantic_scores_minus_1_to_1_normalize_to_0_to_1(self) -> None:
        semantic = [(1, -1.0), (2, -0.5), (3, 0.0), (4, 0.5), (5, 1.0)]
        semantic_rows = [_ranking_chunk(i, f"f{i}.py", "x", "source") for i in range(1, 6)]
        ranked = rank_candidates(
            query="x",
            semantic_matches=semantic,
            semantic_rows=semantic_rows,
            lexical_rows=[],
            options=_options(top=5, diversity=DiversityMode.none, lexical_weight=0.0),
        )
        assert [r.why["semantic_norm"] for r in ranked] == [1.0, 0.75, 0.5, 0.25, 0.0]

    def test_semantic_scores_above_1_use_saturating_transform(self) -> None:
        semantic = [(1, 1.5), (2, 3.0), (3, 10.0)]
        semantic_rows = [_ranking_chunk(i, f"f{i}.py", "x", "source") for i in range(1, 4)]
        ranked = rank_candidates(
            query="x",
            semantic_matches=semantic,
            semantic_rows=semantic_rows,
            lexical_rows=[],
            options=_options(top=3, diversity=DiversityMode.none, lexical_weight=0.0),
        )
        expected = [1.5 / (1.5 + 1.0), 3.0 / (3.0 + 1.0), 10.0 / (10.0 + 1.0)]
        assert [round(r.why["semantic_norm"], 4) for r in ranked] == [round(e, 4) for e in sorted(expected, reverse=True)]  # type: ignore[call-overload]

    def test_negative_weird_semantic_scores_clamp_to_0(self) -> None:
        semantic = [(1, -1.5), (2, -5.0), (3, -100.0)]
        semantic_rows = [_ranking_chunk(i, f"f{i}.py", "x", "source") for i in range(1, 4)]
        ranked = rank_candidates(
            query="x",
            semantic_matches=semantic,
            semantic_rows=semantic_rows,
            lexical_rows=[],
            options=_options(top=3, diversity=DiversityMode.none, lexical_weight=0.0),
        )
        assert all(r.why["semantic_norm"] == 0.0 for r in ranked)  # type: ignore[comparison]


class TestLexicalNormalization:
    def test_lexical_scores_normalize_monotonically(self) -> None:
        lexical_rows = [
            (_ranking_chunk(1, Path("f1.py"), "x", "source"), 0.0),
            (_ranking_chunk(2, Path("f2.py"), "x", "source"), 1.0),
            (_ranking_chunk(3, Path("f3.py"), "x", "source"), 5.0),
            (_ranking_chunk(4, Path("f4.py"), "x", "source"), 100.0),
        ]
        ranked = rank_candidates(
            query="x", semantic_matches=[], semantic_rows=[], lexical_rows=lexical_rows, options=_options(top=4, diversity=DiversityMode.none)
        )
        norms = [r.why["lexical_norm"] for r in ranked]  # type: ignore[assignment]
        assert norms == sorted(norms, reverse=True), f"lexical_norm not monotonic: {norms}"  # type: ignore[type-var]

    def test_lexical_weight_zero_ignores_lexical_contribution(self) -> None:
        semantic = [(1, 0.5)]
        semantic_rows = [_ranking_chunk(1, Path("f1.py"), "x", "source")]
        ranked = rank_candidates(
            query="x",
            semantic_matches=semantic,
            semantic_rows=semantic_rows,
            lexical_rows=[],
            options=_options(lexical_weight=0.0, diversity=DiversityMode.none, lexical_fallback=LexicalFallbackMode.off),
        )
        fused = (1.0 - 0.0) * 0.75 + 0.0 * 0.0
        assert fused * 1.08 == ranked[0].score

    def test_lexical_weight_one_maximizes_lexical_contribution(self) -> None:
        semantic = [(1, 0.5)]
        semantic_rows = [_ranking_chunk(1, Path("f1.py"), "x", "source")]
        lexical_rows = [(_ranking_chunk(1, Path("f1.py"), "x", "source"), 10.0)]
        ranked = rank_candidates(
            query="x",
            semantic_matches=semantic,
            semantic_rows=semantic_rows,
            lexical_rows=lexical_rows,
            options=_options(lexical_weight=1.0, diversity=DiversityMode.none, lexical_fallback=LexicalFallbackMode.off),
        )
        assert ranked[0].why["lexical_norm"] > ranked[0].why["semantic_norm"]  # type: ignore[operator]


class TestPathBoost:
    def test_path_boost_matches_basename(self) -> None:
        semantic = [(1, 0.5), (2, 0.5)]
        semantic_rows = [
            _ranking_chunk(1, Path("src/payments/core.py"), "x", "source"),
            _ranking_chunk(2, Path("src/utils/core.py"), "x", "source"),
        ]
        ranked = rank_candidates(
            query="x",
            semantic_matches=semantic,
            semantic_rows=semantic_rows,
            lexical_rows=[],
            options=_options(path_boosts=(PathBoost(pattern="*core*", weight=0.3),), diversity=DiversityMode.none),
        )
        boosted = next(r for r in ranked if r.why["path_boost"] > 0)  # type: ignore[operator]
        assert "core" in boosted.file_path.name

    def test_path_boost_matches_full_path(self) -> None:
        semantic = [(1, 0.5), (2, 0.5)]
        semantic_rows = [
            _ranking_chunk(1, Path("src/payments/core.py"), "x", "source"),
            _ranking_chunk(2, Path("src/utils/core.py"), "x", "source"),
        ]
        ranked = rank_candidates(
            query="x",
            semantic_matches=semantic,
            semantic_rows=semantic_rows,
            lexical_rows=[],
            options=_options(path_boosts=(PathBoost(pattern="src/payments/*", weight=0.4),), diversity=DiversityMode.none),
        )
        boosted = next(r for r in ranked if r.why["path_boost"] > 0)  # type: ignore[operator]
        assert boosted.file_path == Path("src/payments/core.py")


class TestMinScore:
    def test_min_score_applies_after_boosts_and_multipliers(self) -> None:
        semantic = [(1, 0.8), (2, 0.3)]
        semantic_rows = [
            _ranking_chunk(1, Path("f1.py"), "x", "source"),
            _ranking_chunk(2, Path("f2.py"), "x", "source"),
        ]
        ranked = rank_candidates(
            query="x",
            semantic_matches=semantic,
            semantic_rows=semantic_rows,
            lexical_rows=[],
            options=_options(top=5, min_score=0.5, diversity=DiversityMode.none),
        )
        assert len(ranked) == 1
        assert ranked[0].label == 1


class TestTieBreak:
    def test_tie_break_stable_by_label(self) -> None:
        semantic = [(1, 0.5), (2, 0.5), (3, 0.5)]
        semantic_rows = [_ranking_chunk(i, f"f{i}.py", "x", "source") for i in range(1, 4)]
        ranked = rank_candidates(
            query="x", semantic_matches=semantic, semantic_rows=semantic_rows, lexical_rows=[], options=_options(diversity=DiversityMode.none)
        )
        labels = [r.label for r in ranked]
        assert labels == sorted(labels)


class TestDiversity:
    def test_diversity_file_limits_by_file(self) -> None:
        semantic = [(1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5)]
        semantic_rows = [
            _ranking_chunk(1, Path("a.py"), "x", "source"),
            _ranking_chunk(2, Path("a.py"), "y", "source"),
            _ranking_chunk(3, Path("b.py"), "z", "source"),
            _ranking_chunk(4, Path("c.py"), "w", "source"),
        ]
        ranked = rank_candidates(
            query="x",
            semantic_matches=semantic,
            semantic_rows=semantic_rows,
            lexical_rows=[],
            options=_options(top=3, diversity=DiversityMode.file, lexical_weight=0.0),
        )
        file_paths = [r.file_path for r in ranked]
        assert file_paths.count(Path("a.py")) <= 2

    def test_diversity_package_limits_by_directory(self) -> None:
        semantic = [(1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5)]
        semantic_rows = [
            _ranking_chunk(1, Path("pkg_a/file1.py"), "x", "source"),
            _ranking_chunk(2, Path("pkg_a/file2.py"), "y", "source"),
            _ranking_chunk(3, Path("pkg_b/file3.py"), "z", "source"),
            _ranking_chunk(4, Path("pkg_b/file4.py"), "w", "source"),
        ]
        ranked = rank_candidates(
            query="x",
            semantic_matches=semantic,
            semantic_rows=semantic_rows,
            lexical_rows=[],
            options=_options(top=3, diversity=DiversityMode.package, lexical_weight=0.0),
        )
        packages = {r.file_path.parent for r in ranked}
        assert len(packages) >= 2

    def test_diversity_window_prevents_one_file_from_dominating(self) -> None:
        semantic = [(i, 1.0 - i * 0.01) for i in range(1, 11)]
        semantic_rows = [_ranking_chunk(i, Path("a.py" if i <= 6 else f"b{i}.py"), f"x{i}", "source") for i in range(1, 11)]
        ranked = rank_candidates(
            query="x",
            semantic_matches=semantic,
            semantic_rows=semantic_rows,
            lexical_rows=[],
            options=_options(top=5, diversity=DiversityMode.window, lexical_weight=0.0),
        )
        a_count = sum(1 for r in ranked if r.file_path == Path("a.py"))
        assert a_count <= 2


class TestWhy:
    def test_why_contains_required_fields(self) -> None:
        semantic = [(1, 0.8)]
        semantic_rows = [_ranking_chunk(1, Path("src/a.py"), "rollback payment", "source")]
        ranked = rank_candidates(
            query="rollback payment",
            semantic_matches=semantic,
            semantic_rows=semantic_rows,
            lexical_rows=[],
            options=_options(path_boosts=(PathBoost(pattern="src/*", weight=0.2),)),
        )
        why = ranked[0].why
        assert "semantic_norm" in why
        assert "lexical_norm" in why
        assert "lexical_only" in why
        assert "role_multiplier" in why
        assert "path_boost" in why

    def test_why_query_terms_deduplicated_and_limited(self) -> None:
        semantic = [(1, 0.8)]
        semantic_rows = [_ranking_chunk(1, Path("a.py"), "x", "source")]
        ranked = rank_candidates(
            query="rollback rollback rollback payment payment payment",
            semantic_matches=semantic,
            semantic_rows=semantic_rows,
            lexical_rows=[],
            options=_options(),
        )
        terms = ranked[0].why.get("query_terms", [])  # type: ignore[assignment]
        assert len(terms) <= 8  # type: ignore[arg-type]
        assert len(terms) == len(set(terms))  # type: ignore[arg-type,call-overload]


def test_rank_candidates_fill_keeps_semantic_ahead_of_lexical_only() -> None:
    semantic = [(1, 0.8)]
    semantic_rows = [_ranking_chunk(1, Path("src/service.py"), "rollback payment service", "source")]
    lexical_rows = [(_ranking_chunk(2, Path("docs/rollback.md"), "rollback payment", "docs"), 10.0)]
    ranked = rank_candidates(query="rollback payment", semantic_matches=semantic, semantic_rows=semantic_rows, lexical_rows=lexical_rows, options=_options())
    assert [r.label for r in ranked[:2]] == [1, 2]
    assert ranked[0].why is not None
    assert "role_multiplier" in ranked[0].why


def test_rank_candidates_lexical_fallback_off_removes_lexical_only() -> None:
    ranked = rank_candidates(
        query="q",
        semantic_matches=[],
        semantic_rows=[],
        lexical_rows=[(_ranking_chunk(2, Path("README.md"), "q", "docs"), 3.0)],
        options=_options(lexical_fallback=LexicalFallbackMode.off),
    )
    assert ranked == []


def test_token_coverage_filter_never_drops_semantic_rows() -> None:
    query = "give customers their money back after failed payment"
    semantic = [(1, 0.8)]
    semantic_rows = [_ranking_chunk(1, Path("src/service.py"), "rollback refund", "source")]
    lexical_rows = [(_ranking_chunk(2, Path("docs/x.md"), "unrelated words here", "docs"), 10.0)]
    ranked = rank_candidates(query=query, semantic_matches=semantic, semantic_rows=semantic_rows, lexical_rows=lexical_rows, options=_options())
    assert [r.label for r in ranked] == [1]


def test_rank_candidates_scores_are_distinguishable_and_bounded() -> None:
    semantic = [(1, 0.84), (2, 0.71), (3, 0.58)]
    semantic_rows = [
        _ranking_chunk(1, Path("src/core.py"), "rollback payment orchestration", "source"),
        _ranking_chunk(2, Path("src/payments.py"), "rollback payment handler", "source"),
        _ranking_chunk(3, Path("docs/rollback.md"), "rollback payment guide", "docs"),
    ]
    lexical_rows = [
        (_ranking_chunk(1, Path("src/core.py"), "rollback payment orchestration", "source"), 2.0),
        (_ranking_chunk(2, Path("src/payments.py"), "rollback payment handler", "source"), 6.0),
        (_ranking_chunk(3, Path("docs/rollback.md"), "rollback payment guide", "docs"), 5.0),
    ]

    ranked = rank_candidates(
        query="rollback payment",
        semantic_matches=semantic,
        semantic_rows=semantic_rows,
        lexical_rows=lexical_rows,
        options=_options(top=3, lexical_weight=0.35, diversity=DiversityMode.none, lexical_fallback=LexicalFallbackMode.off),
    )

    assert ranked
    assert all(0.0 <= r.score <= 1.0 for r in ranked)
    assert len({round(r.score, 4) for r in ranked}) >= 2


def test_rank_candidates_path_boost_and_diversity_file() -> None:
    semantic = [(1, 0.7), (2, 0.7), (3, 0.7)]
    semantic_rows = [
        _ranking_chunk(1, Path("src/a.py"), "x", "source"),
        _ranking_chunk(2, Path("src/a.py"), "y", "source"),
        _ranking_chunk(3, Path("tests/a_test.py"), "z", "test"),
    ]
    ranked = rank_candidates(
        query="x",
        semantic_matches=semantic,
        semantic_rows=semantic_rows,
        lexical_rows=[],
        options=_options(top=2, diversity=DiversityMode.file, lexical_top=0, lexical_weight=0.0, path_boosts=(PathBoost(pattern="src/*", weight=0.2),)),
    )
    assert len(ranked) == 2
    assert ranked[0].file_path == Path("src/a.py")


def test_role_multiplier_prefers_source_for_implementation_like_query() -> None:
    ranked = rank_candidates(
        query="implement rollback payment flow",
        semantic_matches=[(1, 0.6), (2, 0.6)],
        semantic_rows=[
            _ranking_chunk(1, Path("src/payments/service.py"), "rollback payment implementation", "source"),
            _ranking_chunk(2, Path("docs/payments.md"), "rollback payment guide", "docs"),
        ],
        lexical_rows=[],
        options=_options(lexical_weight=0.0),
    )
    assert [row.file_role.value for row in ranked[:2]] == ["source", "docs"]


def test_role_multiplier_does_not_break_docs_or_config_or_test_queries() -> None:
    docs_ranked = rank_candidates(
        query="readme setup guide",
        semantic_matches=[(1, 1.0), (2, 0.3)],
        semantic_rows=[
            _ranking_chunk(1, Path("README.md"), "setup guide", "docs"),
            _ranking_chunk(2, Path("src/setup.py"), "setup function", "source"),
        ],
        lexical_rows=[],
        options=_options(lexical_weight=0.0, lexical_fallback=LexicalFallbackMode.off),
    )
    assert docs_ranked[0].file_role.value == "docs"

    config_ranked = rank_candidates(
        query="yaml config timeout",
        semantic_matches=[(3, 1.0), (4, 0.3)],
        semantic_rows=[
            _ranking_chunk(3, Path("config/app.yaml"), "timeout: 30", "config"),
            _ranking_chunk(4, Path("src/config_loader.py"), "load config", "source"),
        ],
        lexical_rows=[],
        options=_options(lexical_weight=0.0, lexical_fallback=LexicalFallbackMode.off),
    )
    assert config_ranked[0].file_role.value == "config"

    test_ranked = rank_candidates(
        query="integration test rollback",
        semantic_matches=[(5, 1.0), (6, 0.3)],
        semantic_rows=[
            _ranking_chunk(5, Path("src/integrationTest/java/com/acme/FlowIT.java"), "rollback integration test", "test"),
            _ranking_chunk(6, Path("src/main/java/com/acme/Flow.java"), "rollback flow", "source"),
        ],
        lexical_rows=[],
        options=_options(lexical_weight=0.0, lexical_fallback=LexicalFallbackMode.off),
    )
    assert test_ranked[0].file_role.value == "test"


def test_rank_candidates_rejects_non_positive_top() -> None:
    ranked = rank_candidates(query="q", semantic_matches=[], semantic_rows=[], lexical_rows=[], options=_options(top=0))

    assert ranked == []


def test_lexical_fallback_empty_mode_zeroes_lexical_only_scores() -> None:
    ranked = rank_candidates(
        query="rollback payment",
        semantic_matches=[(1, 0.9)],
        semantic_rows=[_ranking_chunk(1, Path("src/a.py"), "rollback payment", "source")],
        lexical_rows=[(_ranking_chunk(2, Path("docs/b.md"), "rollback payment", "docs"), 5.0)],
        options=_options(lexical_fallback=LexicalFallbackMode.empty),
    )

    scores = {r.label: r.score for r in ranked}
    assert scores[2] == 0.0
    assert scores[1] > 0.0


def test_diversity_package_caps_hits_per_directory_even_when_top_exceeds_supply() -> None:
    rows = [_ranking_chunk(i, f"pkg/f{i}.py", f"t{i}", "source") for i in range(1, 5)]
    ranked = rank_candidates(
        query="t",
        semantic_matches=[(i, 1.0 - i * 0.05) for i in range(1, 5)],
        semantic_rows=rows,
        lexical_rows=[],
        options=_options(top=4, diversity=DiversityMode.package, lexical_weight=0.0),
    )

    assert [r.label for r in ranked] == [1, 2]


def test_detail_rows_without_usearch_label_or_known_role_degrade_gracefully() -> None:
    ranked = rank_candidates(
        query="needle",
        semantic_matches=[(7, 0.8)],
        semantic_rows=[_ranking_chunk(7, Path("src/c.py"), "needle", "bogus_role")],
        lexical_rows=[],
        options=_options(),
    )

    assert [r.label for r in ranked] == [7]
    assert ranked[0].file_role == FileRole.unknown
