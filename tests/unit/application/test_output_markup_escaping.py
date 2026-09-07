from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from simgrep.models import FileRole, RenderOptions, ResultFormat, SearchResult
from simgrep.output import render_search_results


def _result(
    tmp_path: Path,
    chunk_text: str,
    *,
    line_start: int = 1,
    line_end: int = 1,
    start_char: int | None = None,
) -> SearchResult:
    return SearchResult(
        label=0,
        score=0.9,
        file_path=tmp_path / "a.txt",
        chunk_text=chunk_text,
        start_char=0 if start_char is None else start_char,
        end_char=len(chunk_text),
        line_start=line_start,
        line_end=line_end,
        file_role=FileRole.source,
        language="txt",
    )


def _render(tmp_path: Path, chunk_text: str, *, fmt: ResultFormat) -> str:
    buffer = io.StringIO()
    console = Console(file=buffer, width=400)
    options = RenderOptions(format=fmt)
    render_search_results([_result(tmp_path, chunk_text)], options=options, console=console)
    return buffer.getvalue()


class TestRichOutputPreservesBrackets:
    def test_bracketed_chunk_text_renders_verbatim(self, tmp_path: Path) -> None:
        output = _render(tmp_path, "  a[b]c", fmt=ResultFormat.rich)
        assert "  a[b]c" in output

    def test_closing_bracket_tag_does_not_raise(self, tmp_path: Path) -> None:
        output = _render(tmp_path, "  x[/usr/bin]y", fmt=ResultFormat.rich)
        assert "  x[/usr/bin]y" in output

    def test_svg_style_tag_survives(self, tmp_path: Path) -> None:
        output = _render(tmp_path, "[svg] drawing", fmt=ResultFormat.rich)
        assert "[svg] drawing" in output

    def test_context_line_brackets_render_verbatim(self, tmp_path: Path) -> None:
        result = SearchResult(
            label=0,
            score=0.9,
            file_path=tmp_path / "a.txt",
            chunk_text="match [b]",
            start_char=0,
            end_char=9,
            line_start=2,
            line_end=2,
            file_role=FileRole.source,
            language="txt",
        )
        buffer = io.StringIO()
        console = Console(file=buffer, width=400)
        options = RenderOptions(format=ResultFormat.rich, context_lines=1)
        render_search_results([result], options=options, console=console)
        assert "[b]" in buffer.getvalue()


class TestCompactOutputPreservesBrackets:
    def test_compact_keeps_bracketed_snippet(self, tmp_path: Path) -> None:
        output = _render(tmp_path, "value[0] = x", fmt=ResultFormat.compact)
        assert "value[0] = x" in output

    def test_compact_closing_bracket_does_not_raise(self, tmp_path: Path) -> None:
        output = _render(tmp_path, "see [/etc/hosts] file", fmt=ResultFormat.compact)
        assert "see [/etc/hosts] file" in output
