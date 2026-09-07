"""Unit tests for the `simgrep similar` flow primitives: SOURCE resolution,
combined two-anchor scoring, and self-exclusion span logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from simgrep.corpus import StoredChunk
from simgrep.errors import SearchError
from simgrep.models import SearchOptions
from simgrep.ranking import rank_candidates
from simgrep.search import combine_candidate_scores, filter_self_matches, resolve_anchor
from tests.conftest import _ranking_chunk


def _detail(label: int, file_path: Path | str, start_char: int, end_char: int) -> StoredChunk:
    return _ranking_chunk(label, file_path, text="x", start_char=start_char, end_char=end_char)


class TestResolveSourceForms:
    def test_dash_reads_stdin(self) -> None:
        anchor = resolve_anchor("-", stdin_text="piped anchor text\n")
        assert anchor.text == "piped anchor text\n"
        assert anchor.origin is None

    def test_dash_without_stdin_fails(self) -> None:
        with pytest.raises(SearchError, match="piped stdin"):
            resolve_anchor("-")

    def test_at_file_reads_full_content(self, tmp_path: Path) -> None:
        target = tmp_path / "anchor.py"
        target.write_text("alpha\nbeta\n", encoding="utf-8")
        anchor = resolve_anchor(f"@{target}")
        assert anchor.text == "alpha\nbeta\n"
        assert anchor.origin is not None and anchor.origin.is_absolute()
        assert (anchor.start_char, anchor.end_char) == (0, len("alpha\nbeta\n"))

    def test_at_file_missing_fails(self, tmp_path: Path) -> None:
        with pytest.raises(SearchError):
            resolve_anchor(f"@{tmp_path}/missing.py")

    def test_line_range_resolves_lines_and_span(self, tmp_path: Path) -> None:
        target = tmp_path / "range.txt"
        target.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
        anchor = resolve_anchor(f"{target}:2-3")
        assert anchor.text == "two\nthree\n"
        assert anchor.origin == target.absolute()
        # chars of "one\n" prefix up to end of "three"
        assert (anchor.start_char, anchor.end_char) == (4, 14)

    def test_line_range_clamps_end_beyond_eof(self, tmp_path: Path) -> None:
        target = tmp_path / "short.txt"
        target.write_text("only line\n", encoding="utf-8")
        anchor = resolve_anchor(f"{target}:1-99")
        assert anchor.text == "only line\n"
        assert (anchor.start_char, anchor.end_char) == (0, len("only line\n"))

    def test_line_range_start_beyond_eof_errors(self, tmp_path: Path) -> None:
        target = tmp_path / "short.txt"
        target.write_text("only line\n", encoding="utf-8")
        with pytest.raises(SearchError):
            resolve_anchor(f"{target}:9-10")

    def test_line_range_inverted_errors(self, tmp_path: Path) -> None:
        target = tmp_path / "short.txt"
        target.write_text("only line\n", encoding="utf-8")
        with pytest.raises(SearchError):
            resolve_anchor(f"{target}:3-1")

    def test_regex_with_existing_file_wins_over_inline(self, tmp_path: Path) -> None:
        target = tmp_path / "weird: name.txt"
        target.write_text("inside\n", encoding="utf-8")
        anchor = resolve_anchor(f"{target}:1-1")
        assert anchor.text == "inside\n"

    def test_colon_literal_without_file_is_inline(self) -> None:
        anchor = resolve_anchor("note: this is literal text")
        assert anchor.text == "note: this is literal text"
        assert anchor.origin is None

    def test_empty_stdin_errors(self) -> None:
        with pytest.raises(SearchError):
            resolve_anchor("-", stdin_text="   \n")

    def test_empty_file_errors(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.txt"
        target.write_text("", encoding="utf-8")
        with pytest.raises(SearchError):
            resolve_anchor(f"@{target}")

    def test_inline_whitespace_only_errors(self) -> None:
        with pytest.raises(SearchError):
            resolve_anchor("   ")

    def test_single_line_anchor_resolves_span(self, tmp_path: Path) -> None:
        target = tmp_path / "greet.py"
        target.write_text("one\ntwo\nthree\n", encoding="utf-8")
        anchor = resolve_anchor(f"{target}:2")
        assert anchor.origin == target.absolute()
        assert anchor.text == "two\n"

    def test_at_single_line_anchor_resolves_span(self, tmp_path: Path) -> None:
        target = tmp_path / "greet.py"
        target.write_text("one\ntwo\nthree\n", encoding="utf-8")
        anchor = resolve_anchor(f"@{target}:3")
        assert anchor.origin == target.absolute()
        assert anchor.text == "three\n"

    def test_file_like_path_line_without_file_errors(self) -> None:
        with pytest.raises(SearchError, match="not found"):
            resolve_anchor("greet.py:6")

    def test_spaced_colon_digits_stay_literal(self) -> None:
        anchor = resolve_anchor("ratio 3:5")
        assert anchor.text == "ratio 3:5"
        assert anchor.origin is None

    def test_bare_number_colon_digits_stay_literal(self) -> None:
        anchor = resolve_anchor("12:30")
        assert anchor.text == "12:30"
        assert anchor.origin is None


class TestAnchorFileDecoding:
    """Anchor bytes must decode exactly like the indexer (utf-8-sig, no newline
    translation) so char spans reconcile with stored chunk offsets."""

    def test_crlf_preserved_no_newline_translation(self, tmp_path: Path) -> None:
        target = tmp_path / "crlf.py"
        raw = b"alpha\r\nbeta\r\n"
        target.write_bytes(raw)
        anchor = resolve_anchor(f"@{target}")
        assert anchor.text == raw.decode("utf-8")
        assert (anchor.start_char, anchor.end_char) == (0, len(raw))

    def test_bom_stripped_and_span_bom_less(self, tmp_path: Path) -> None:
        target = tmp_path / "bom.py"
        target.write_bytes(b"\xef\xbb\xbfalpha\nbeta\n")
        anchor = resolve_anchor(f"@{target}")
        assert anchor.text == "alpha\nbeta\n"
        assert (anchor.start_char, anchor.end_char) == (0, len("alpha\nbeta\n"))

    def test_line_range_offsets_computed_over_decoded_text(self, tmp_path: Path) -> None:
        target = tmp_path / "mixed.py"
        target.write_bytes(b"\xef\xbb\xbffirst\r\nsecond line\r\n")
        anchor = resolve_anchor(f"{target}:2-2")
        assert anchor.text == "second line\r\n"
        assert (anchor.start_char, anchor.end_char) == (7, 20)

    def test_non_utf8_file_raises_search_error(self, tmp_path: Path) -> None:
        target = tmp_path / "latin1.py"
        target.write_bytes(b"caf\xe9 option\n")
        with pytest.raises(SearchError):
            resolve_anchor(f"@{target}")


class TestCombineCandidateScores:
    def test_combined_score_subtracts_lambda_term(self) -> None:
        combined = dict(combine_candidate_scores({1: 0.9, 2: 0.8}, {2: 0.5}, 0.5))
        assert combined[1] == pytest.approx(0.9)
        assert combined[2] == pytest.approx(0.8 - 0.25)

    def test_missing_side_contributes_zero(self) -> None:
        combined = dict(combine_candidate_scores({1: 0.9}, {2: 0.7}, 1.0))
        assert combined[1] == pytest.approx(0.9)
        assert combined[2] == pytest.approx(-0.7)

    def test_union_sorted_descending(self) -> None:
        pairs = combine_candidate_scores({1: 0.5, 2: 0.6}, {3: 0.1}, 0.5)
        scores = dict(pairs)
        assert [label for label, _ in pairs] == [2, 1, 3]
        assert scores[2] == pytest.approx(0.6)
        assert scores[1] == pytest.approx(0.5)
        assert scores[3] == pytest.approx(-0.05)

    def test_no_unlike_returns_like_scores(self) -> None:
        pairs = combine_candidate_scores({1: 0.9}, {}, 0.5)
        assert pairs == [(1, pytest.approx(0.9))]


class TestSelfExclusion:
    def test_same_file_overlap_dropped(self, tmp_path: Path) -> None:
        origin = tmp_path / "a.py"
        kept = filter_self_matches([_detail(1, origin, 0, 100)], origin, 10, 20, include_self=False)
        assert kept == []

    def test_other_file_kept_even_with_matching_span(self, tmp_path: Path) -> None:
        kept = filter_self_matches([_detail(1, tmp_path / "b.py", 10, 20)], tmp_path / "a.py", 10, 20, include_self=False)
        assert len(kept) == 1

    def test_adjacent_span_not_overlapping(self, tmp_path: Path) -> None:
        origin = tmp_path / "a.py"
        kept = filter_self_matches([_detail(1, origin, 20, 30)], origin, 0, 20, include_self=False)
        assert len(kept) == 1

    def test_include_self_bypass(self, tmp_path: Path) -> None:
        origin = tmp_path / "a.py"
        kept = filter_self_matches([_detail(1, origin, 0, 100)], origin, 10, 20, include_self=True)
        assert len(kept) == 1

    def test_relative_result_path_resolved_against_base(self, tmp_path: Path) -> None:
        kept = filter_self_matches([_detail(1, Path("a.py"), 0, 100)], tmp_path / "a.py", 0, 5, include_self=False, base_path=tmp_path)
        assert kept == []


class TestWhyContrastiveBreakdown:
    @staticmethod
    def _options() -> SearchOptions:
        return SearchOptions(query="retry request helper pattern", top=3, lexical_top=0, lexical_weight=0.0)

    def test_semantic_like_reports_true_like_score(self) -> None:
        rows = [_ranking_chunk(1, "src/x.py", "retry request helper pattern", "source")]
        results = rank_candidates(
            query=self._options().query,
            semantic_matches=[(1, 0.45)],
            semantic_rows=rows,
            lexical_rows=[],
            options=self._options(),
            contrast_unlike={1: 0.45},
            contrast_like={1: 0.9},
        )
        why = results[0].why
        assert why["semantic_like"] == pytest.approx(0.9)
        assert why["semantic_unlike"] == pytest.approx(0.45)

    def test_default_without_contrast_like_keeps_legacy_combined_value(self) -> None:
        rows = [_ranking_chunk(1, "src/x.py", "retry request helper pattern", "source")]
        results = rank_candidates(
            query=self._options().query,
            semantic_matches=[(1, 0.45)],
            semantic_rows=rows,
            lexical_rows=[],
            options=self._options(),
            contrast_unlike={1: 0.45},
        )
        assert results[0].why["semantic_like"] == pytest.approx(0.45)


class TestTokenCoverageSemanticSurvival:
    def test_semantic_rows_survive_coverage_filter_with_long_anchor(self) -> None:
        query = "give customers their money back after failed payment"
        options = SearchOptions(query=query, top=3, lexical_top=10, lexical_weight=0.25)
        semantic_rows = [_ranking_chunk(1, "src/service.py", "rollback refund", "source")]
        lexical_rows = [(_ranking_chunk(2, "docs/x.md", "totally unrelated prose", "docs"), 10.0)]
        results = rank_candidates(query=query, semantic_matches=[(1, 0.8)], semantic_rows=semantic_rows, lexical_rows=lexical_rows, options=options)
        assert [r.label for r in results] == [1]
