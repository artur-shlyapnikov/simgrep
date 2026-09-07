"""Regression tests: grep format must keep true line numbers under --no-line-numbers."""

from pathlib import Path

from simgrep.models import FileRole, RenderOptions, ResultFormat, SearchResult
from simgrep.output import enrich_result, format_compact, format_grep, format_json


def _result(line_start: int = 42) -> SearchResult:
    return SearchResult(
        label=0,
        score=1.0,
        file_path=Path("f.txt"),
        chunk_text="hello",
        start_char=0,
        end_char=5,
        line_start=line_start,
        line_end=line_start,
        file_role=FileRole.source,
        language="text",
    )


def test_grep_format_keeps_true_line_numbers_with_show_line_numbers_false() -> None:
    options = RenderOptions(format=ResultFormat.grep, show_line_numbers=False, context_lines=0)
    display = enrich_result(_result(42), options)
    out = format_grep(display, show_scores=False)
    assert ":42:" in out


def test_json_still_omits_line_numbers_with_show_line_numbers_false() -> None:
    options = RenderOptions(format=ResultFormat.json, show_line_numbers=False, context_lines=0)
    display = enrich_result(_result(42), options)
    record = format_json([display], show_scores=False, show_why=False)
    assert "line_start" not in record
    assert "line_end" not in record


def test_compact_hides_line_numbers_with_show_line_numbers_false() -> None:
    options = RenderOptions(format=ResultFormat.compact, show_line_numbers=False, show_scores=False, context_lines=0)
    display = enrich_result(_result(42), options)
    out = format_compact(display, show_scores=False, show_line_numbers=False)
    assert ":42" not in out.split("  ")[0]
