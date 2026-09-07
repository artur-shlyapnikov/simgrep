from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from rich.console import Console

from simgrep.models import FileRole, RenderOptions, ResultFormat, SearchResult
from simgrep.output import enrich_result, format_grep, render_search_results


def _result(path: Path, *, score: float = 0.9, why: dict[str, object] | None = None) -> SearchResult:
    return SearchResult(
        label=0,
        score=score,
        file_path=path,
        chunk_text="needle",
        start_char=0,
        end_char=6,
        line_start=None,
        line_end=None,
        file_role=FileRole.source,
        language="py",
        why=why or {},
    )


def test_format_grep_flattens_multiline_snippet_and_toggles_scores(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("needle here\nsecond line\n")
    result = SearchResult(
        label=0,
        score=0.9,
        file_path=f,
        chunk_text="needle here\nsecond line",
        start_char=0,
        end_char=23,
        line_start=None,
        line_end=None,
        file_role=FileRole.source,
        language="txt",
    )
    display = enrich_result(result, RenderOptions(format=ResultFormat.grep))

    scored = format_grep(display, show_scores=True)
    plain = format_grep(display, show_scores=False)

    assert "\n" not in scored and "\n" not in plain
    assert scored.endswith(":1:0.900: needle here second line")
    assert plain.endswith(":1: needle here second line")


@pytest.mark.parametrize(
    ("fmt", "expected_stdout"),
    [
        (ResultFormat.count, "0\n"),
        (ResultFormat.json, "[]\n"),
        (ResultFormat.jsonl, ""),
        (ResultFormat.grep, ""),
    ],
)
def test_render_machine_formats_with_no_results_print_machine_output(
    fmt: ResultFormat, expected_stdout: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    render_search_results([], options=RenderOptions(format=fmt, base_path=tmp_path))

    assert capsys.readouterr().out == expected_stdout


def test_render_paths_sorts_relative_paths_and_keeps_absolute_outside_base(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    proj = tmp_path / "proj"
    (proj / "src").mkdir(parents=True)
    inside = proj / "src" / "b.py"
    inside.write_text("x")
    outside = tmp_path / "elsewhere.py"
    outside.write_text("x")

    render_search_results([_result(inside), _result(outside)], options=RenderOptions(format=ResultFormat.paths, base_path=proj))

    assert capsys.readouterr().out == f"{outside.resolve()}\nsrc/b.py\n"


def test_render_populated_json_jsonl_and_grep_emit_records(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "a.txt"
    f.write_text("needle")
    result = _result(f, why={"semantic_norm": 0.9})

    render_search_results([result], options=RenderOptions(format=ResultFormat.json, show_why=True, base_path=tmp_path))
    records = json.loads(capsys.readouterr().out)
    assert len(records) == 1
    assert records[0]["path"] == "a.txt"
    assert records[0]["why"] == {"semantic_norm": 0.9}
    assert records[0]["stale_offsets"] is False

    render_search_results([result], options=RenderOptions(format=ResultFormat.jsonl, base_path=tmp_path))
    out = capsys.readouterr().out
    assert len(out.strip().splitlines()) == 1
    assert json.loads(out)["stale_offsets"] is False

    render_search_results([result], options=RenderOptions(format=ResultFormat.grep, show_scores=True, base_path=tmp_path))
    assert capsys.readouterr().out == "a.txt:1:0.900: needle\n"


def test_render_rich_shows_line_range_context_and_why(tmp_path: Path) -> None:
    f = tmp_path / "code.py"
    f.write_text("import os\nneedle here\nafter one\nafter two\n")
    result = SearchResult(
        label=0,
        score=0.9,
        file_path=f,
        chunk_text="needle here\n",
        start_char=9,
        end_char=21,
        line_start=None,
        line_end=None,
        file_role=FileRole.source,
        language="py",
        why={"semantic_norm": 0.95},
    )
    buffer = io.StringIO()
    console = Console(file=buffer, width=240)

    render_search_results(
        [result],
        options=RenderOptions(format=ResultFormat.rich, context_lines=1, show_why=True),
        console=console,
    )

    text = buffer.getvalue()
    assert text.startswith("Search Results (1):\n")
    assert ":1-2" in text  # multi-line span range prefix (line 181 branch)
    assert "import os" in text  # context_before rendering
    assert "after one" in text  # context_after rendering
    assert "why: semantic_norm=0.95" in text  # dim why footer, filtered key formatting


def test_render_rich_indents_every_snippet_line(tmp_path: Path) -> None:
    f = tmp_path / "multi.py"
    f.write_text("first\ndef two():\n    pass\n")
    result = SearchResult(
        label=0,
        score=0.9,
        file_path=f,
        chunk_text="def one():\n    pass\ndef two():\n    pass",
        start_char=0,
        end_char=36,
        line_start=None,
        line_end=None,
        file_role=FileRole.source,
        language="py",
        why={},
    )
    buffer = io.StringIO()
    console = Console(file=buffer, width=240)

    render_search_results([result], options=RenderOptions(format=ResultFormat.rich), console=console)

    snippet_lines = [line for line in buffer.getvalue().splitlines() if "def " in line or line.strip() == "pass"]
    assert snippet_lines, "expected snippet lines in output"
    for line in snippet_lines:
        assert line.startswith("  "), f"snippet line not indented: {line!r}"
