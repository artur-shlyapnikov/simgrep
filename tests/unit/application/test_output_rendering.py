from __future__ import annotations

import json
from pathlib import Path

from simgrep.models import DisplaySearchResult, FileRole, RenderOptions, ResultFormat, SearchResult
from simgrep.output import (
    _truncate,
    enrich_result,
    format_compact,
    format_json,
    format_jsonl,
    format_paths,
)


def _result(
    tmp_path: Path,
    chunk_text: str = "hello world",
    start_char: int = 0,
    end_char: int = 11,
    line_start: int = 1,
    line_end: int = 1,
) -> SearchResult:
    return SearchResult(
        label=0,
        score=0.9,
        file_path=tmp_path / "a.txt",
        chunk_text=chunk_text,
        start_char=start_char,
        end_char=end_char,
        line_start=line_start,
        line_end=line_end,
        file_role=FileRole.source,
        language="txt",
    )


class TestEnrichResultLineRecalc:
    def test_enrich_recalculates_lines_when_context_rendering(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("line one\nline two match\nline three")

        result = _result(tmp_path, chunk_text="line two match", start_char=10, end_char=23, line_start=1, line_end=1)

        options = RenderOptions(format=ResultFormat.json, context_lines=1)
        display = enrich_result(result, options)

        assert display.line_start == 2
        assert display.line_end == 2
        assert "match" in display.snippet
        assert len(display.context_before) == 1
        assert len(display.context_after) == 1

    def test_enrich_preserves_original_when_no_context(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("line one\nline two match\nline three")

        result = _result(tmp_path, chunk_text="line two match", start_char=10, end_char=23, line_start=1, line_end=1)

        options = RenderOptions(format=ResultFormat.json, context_lines=0)
        display = enrich_result(result, options)

        assert display.line_start == 1
        assert display.line_end == 1


class TestEnrichResultStaleOffsets:
    def test_stale_when_offsets_beyond_file_length(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("short")

        result = _result(tmp_path, start_char=0, end_char=100, line_start=1, line_end=1)

        options = RenderOptions(format=ResultFormat.json, context_lines=1)
        display = enrich_result(result, options)

        assert display.stale_offsets is True

    def test_stale_when_file_disappears(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("hello world")
        result = _result(tmp_path)
        f.unlink()

        options = RenderOptions(format=ResultFormat.json, context_lines=1)
        display = enrich_result(result, options)

        assert display.stale_offsets is True

    def test_no_crash_when_file_missing(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("hello world")
        result = _result(tmp_path)
        f.unlink()

        options = RenderOptions(format=ResultFormat.json, context_lines=1)
        display = enrich_result(result, options)

        assert display.snippet == "hello world"
        assert display.stale_offsets is True


class TestEnrichResultLatin1Fallback:
    def test_latin1_fallback_when_utf8_fails(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_bytes(b"caf\xe9\r\nline two match\r\nline three")

        result = SearchResult(
            label=0,
            score=0.9,
            file_path=f,
            chunk_text="line two match",
            start_char=6,
            end_char=20,
            line_start=1,
            line_end=1,
            file_role=FileRole.source,
            language="txt",
        )

        options = RenderOptions(format=ResultFormat.json, context_lines=1)
        display = enrich_result(result, options)

        assert display.line_start == 2
        assert display.line_end == 2
        assert "match" in display.snippet
        assert display.stale_offsets is False


class TestTruncate:
    def test_truncate_max_chars_none(self) -> None:
        text = "hello world"
        assert _truncate(text, None) == text

    def test_truncate_under_limit(self) -> None:
        text = "hello"
        assert _truncate(text, 10) == text

    def test_truncate_at_limit(self) -> None:
        text = "hello"
        assert _truncate(text, 5) == text

    def test_truncate_max_chars_3(self) -> None:
        text = "hello"
        result = _truncate(text, 3)
        assert result == "hel"
        assert len(result) == 3

    def test_truncate_max_chars_2(self) -> None:
        text = "hello"
        result = _truncate(text, 2)
        assert result == "he"

    def test_truncate_max_chars_1(self) -> None:
        text = "hello"
        result = _truncate(text, 1)
        assert result == "h"

    def test_truncate_uses_word_boundary(self) -> None:
        text = "word word word word word"
        result = _truncate(text, 12)
        assert result.endswith("...")
        assert len(result) <= 12
        assert result != ""

    def test_truncate_word_boundary_preserves_at_least_one_word(self) -> None:
        text = "word word word word word"
        result = _truncate(text, 9)
        assert result.endswith("...")
        assert len(result) <= 9
        assert " " not in result or result.strip() == result

    def test_truncate_preserves_snippet_not_empty(self) -> None:
        text = "word " * 100
        result = _truncate(text, 5)
        assert len(result) <= 5
        assert result
        assert result != ""


class TestFormatCompact:
    def test_multiline_shows_first_line_only(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("line one\nline two match\nline three")

        display = DisplaySearchResult(
            search_result=_result(tmp_path, chunk_text="line one\nline two match\nline three", start_char=0, end_char=35, line_start=1, line_end=3),
            display_path="/fake/a.txt",
            line_start=1,
            line_end=3,
            snippet="line one\nline two match\nline three",
            context_before=(),
            context_after=(),
            stale_offsets=False,
        )

        formatted = format_compact(display, show_scores=False, show_line_numbers=True)
        lines = formatted.splitlines()
        assert len(lines) == 1
        assert "line one" in lines[0]
        assert "line two" not in lines[0]


class TestFormatJson:
    def test_json_includes_required_fields(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("hello world")

        display = DisplaySearchResult(
            search_result=_result(tmp_path),
            display_path="/fake/a.txt",
            line_start=1,
            line_end=1,
            snippet="hello",
            context_before=(),
            context_after=(),
            stale_offsets=False,
        )

        text = format_json([display], show_scores=True, show_why=True)
        record = json.loads(text)[0]

        assert "path" in record
        assert "score" in record
        assert "line_start" in record
        assert "line_end" in record
        assert "start_char" in record
        assert "end_char" in record
        assert "text" in record
        assert "stale_offsets" in record

    def test_json_why_only_when_enabled(self, tmp_path: Path) -> None:
        result = SearchResult(
            label=0,
            score=0.9,
            file_path=tmp_path / "a.txt",
            chunk_text="hello",
            start_char=0,
            end_char=5,
            line_start=1,
            line_end=1,
            file_role=FileRole.source,
            language="txt",
            why={"term_match": ["hello"]},
        )

        display = DisplaySearchResult(
            search_result=result,
            display_path="/fake/a.txt",
            line_start=1,
            line_end=1,
            snippet="hello",
            context_before=(),
            context_after=(),
            stale_offsets=False,
        )

        text_with = format_json([display], show_scores=True, show_why=True)
        text_without = format_json([display], show_scores=True, show_why=False)

        assert "why" in json.loads(text_with)[0]
        assert "why" not in json.loads(text_without)[0]


class TestFormatPathsDeduplication:
    def test_paths_deduplicated_for_multiple_chunks_same_file(self, tmp_path: Path) -> None:
        results = [
            SearchResult(
                label=0,
                score=0.9,
                file_path=tmp_path / "a.txt",
                chunk_text="chunk one",
                start_char=0,
                end_char=9,
                line_start=1,
                line_end=1,
                file_role=FileRole.source,
                language="txt",
            ),
            SearchResult(
                label=1,
                score=0.8,
                file_path=tmp_path / "a.txt",
                chunk_text="chunk two",
                start_char=10,
                end_char=19,
                line_start=2,
                line_end=2,
                file_role=FileRole.source,
                language="txt",
            ),
        ]

        options = RenderOptions(format=ResultFormat.paths)
        text = format_paths(results, options)
        lines = [line for line in text.splitlines() if line.strip()]
        deduped = set(lines)

        assert len(lines) == len(deduped), f"Expected deduplicated, got {lines}"
        assert len(lines) == 1


class TestJsonl:
    def test_jsonl_includes_required_fields(self, tmp_path: Path) -> None:
        display = DisplaySearchResult(
            search_result=_result(tmp_path),
            display_path="/fake/a.txt",
            line_start=1,
            line_end=1,
            snippet="hello",
            context_before=(),
            context_after=(),
            stale_offsets=False,
        )

        text = format_jsonl([display], show_scores=True, show_why=True)
        record = json.loads(text)

        assert "path" in record
        assert "score" in record
        assert "text" in record
        assert "stale_offsets" in record
