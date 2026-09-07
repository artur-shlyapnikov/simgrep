"""Regression tests: context lines must appear in json/jsonl output."""

from pathlib import Path

from simgrep.models import FileRole, SearchResult
from simgrep.records import DisplaySearchResult, display_record


def _display(context_before: tuple[str, ...] = (), context_after: tuple[str, ...] = ()) -> DisplaySearchResult:
    search_result = SearchResult(
        label=0,
        score=1.0,
        file_path=Path("f.txt"),
        chunk_text="hello",
        start_char=0,
        end_char=5,
        line_start=1,
        line_end=1,
        file_role=FileRole.source,
        language="text",
    )
    return DisplaySearchResult(
        search_result=search_result,
        display_path="f.txt",
        line_start=1,
        line_end=1,
        snippet="hello",
        context_before=context_before,
        context_after=context_after,
        stale_offsets=False,
    )


def test_json_record_includes_context_when_present() -> None:
    record = display_record(
        _display(("prev line",), ("next line", "another")),
        show_scores=True,
        show_why=False,
    )
    assert record["context_before"] == ["prev line"]
    assert record["context_after"] == ["next line", "another"]


def test_json_record_omits_context_keys_when_empty() -> None:
    record = display_record(_display(), show_scores=True, show_why=False)
    assert "context_before" not in record
    assert "context_after" not in record


def test_jsonl_includes_context() -> None:
    from simgrep.output import format_jsonl

    out = format_jsonl([_display(("b",), ("a",))], show_scores=True, show_why=False)
    assert '"context_before": ["b"]' in out
    assert '"context_after": ["a"]' in out
