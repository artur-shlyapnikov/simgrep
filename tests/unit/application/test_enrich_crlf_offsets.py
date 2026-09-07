from pathlib import Path

from simgrep.models import FileRole, RenderOptions, ResultFormat, SearchResult
from simgrep.output import enrich_result

CRLF_TEXT = "line1\r\nline2\r\ntarget line3\r\nline4\r\n"
# Offsets of "target line3" computed on the RAW CRLF text, matching what the indexer stores.
CRLF_START = CRLF_TEXT.index("target line3")
CRLF_END = CRLF_START + len("target line3")


def _result(path: Path, chunk_text: str, start_char: int, end_char: int) -> SearchResult:
    return SearchResult(
        label=0,
        score=1.0,
        file_path=path,
        chunk_text=chunk_text,
        start_char=start_char,
        end_char=end_char,
        line_start=None,
        line_end=None,
        file_role=FileRole.source,
        language="text",
    )


def _options() -> RenderOptions:
    return RenderOptions(format=ResultFormat.json, context_lines=1)


def test_enrich_maps_raw_offsets_on_crlf_file(tmp_path: Path) -> None:
    path = tmp_path / "crlf.txt"
    path.write_bytes(CRLF_TEXT.encode("utf-8"))
    result = _result(path, "target line3", CRLF_START, CRLF_END)

    display = enrich_result(result, _options())

    assert display.stale_offsets is False
    assert display.line_start == 3
    assert display.line_end == 3
    assert display.snippet == "target line3\n"
    assert display.context_before == ("line2\n",)
    assert display.context_after == ("line4\n",)


def test_enrich_keeps_lf_mapping_unchanged(tmp_path: Path) -> None:
    lf_text = CRLF_TEXT.replace("\r\n", "\n")
    path = tmp_path / "lf.txt"
    path.write_bytes(lf_text.encode("utf-8"))
    start = lf_text.index("target line3")
    result = _result(path, "target line3", start, start + len("target line3"))

    display = enrich_result(result, _options())

    assert display.stale_offsets is False
    assert display.line_start == 3
    assert display.line_end == 3
    assert display.snippet == "target line3\n"
    assert display.context_before == ("line2\n",)
    assert display.context_after == ("line4\n",)


def test_enrich_does_not_mark_eof_match_stale_on_crlf(tmp_path: Path) -> None:
    # Match ends exactly at EOF of the raw CRLF text; the old normalized-length
    # comparison flagged this fresh result as stale.
    text = "a\r\nb\r\ntail"
    path = tmp_path / "eof.txt"
    path.write_bytes(text.encode("utf-8"))
    start = text.index("tail")
    result = _result(path, "tail", start, len(text))

    display = enrich_result(result, _options())

    assert display.stale_offsets is False
    assert display.line_start == 3
    assert display.line_end == 3


def test_enrich_maps_mixed_line_endings_exactly(tmp_path: Path) -> None:
    # \r\n and \n interleaved: raw offsets must land on line 3 regardless of ending style,
    # and every emitted string (snippet, before, after) is \r\n-normalized.
    text = "alpha\r\nbeta\ntarget line\r\ngamma"
    path = tmp_path / "mixed.txt"
    path.write_bytes(text.encode("utf-8"))
    result = _result(path, "target line", text.index("target line"), text.index("target line") + len("target line"))

    display = enrich_result(result, _options())

    assert display.stale_offsets is False
    assert display.line_start == 3
    assert display.line_end == 3
    assert display.snippet == "target line\n"
    assert display.context_before == ("beta\n",)
    assert display.context_after == ("gamma",)


def test_enrich_stale_beyond_raw_length_falls_back_to_chunk_text(tmp_path: Path) -> None:
    # Offsets past RAW EOF: stale flag set, stored chunk_text returned verbatim (its own
    # \r\n untouched), and the incoming line numbers pass through unmodified.
    text = "keep\r\nthis\r\n"
    path = tmp_path / "stale.txt"
    path.write_bytes(text.encode("utf-8"))
    result = _result(path, "old chunk\r\ntext", 0, len(text) + 50)
    result = result.__class__(**{**result.__dict__, "line_start": 4, "line_end": 9})

    display = enrich_result(result, _options())

    assert display.stale_offsets is True
    assert display.snippet == "old chunk\r\ntext"
    assert display.line_start == 4
    assert display.line_end == 9


def test_enrich_latin1_fallback_with_crlf_exact_values(tmp_path: Path) -> None:
    # Byte-reading latin-1 fallback: non-UTF8 prefix must not shift offsets, and the
    # decoded CRLF lines come back normalized. Stronger sibling of the smoke assertion
    # living in test_output_rendering.py (which we must not edit).
    path = tmp_path / "latin_crlf.txt"
    path.write_bytes(b"caf\xe9\r\nmatch me\r\nfin")
    result = _result(path, "match me", 6, 14)

    display = enrich_result(result, _options())

    assert display.stale_offsets is False
    assert display.line_start == 2
    assert display.line_end == 2
    assert display.snippet == "match me\n"
    assert display.context_before == ("café\n",)
    assert display.context_after == ("fin",)


def test_enrich_context_window_clamps_at_file_edges_on_crlf(tmp_path: Path) -> None:
    # Pins the exact edge contract: no before-lines at BOF, and past-EOF context slots
    # surface as "" padding (not trimmed) — see design doc Q3.
    text = "one\r\ntwo\r\n"
    path = tmp_path / "edges.txt"
    path.write_bytes(text.encode("utf-8"))
    options = RenderOptions(format=ResultFormat.json, context_lines=5)

    first = enrich_result(_result(path, "one", 0, 3), options)
    last = enrich_result(_result(path, "two", text.index("two"), text.index("two") + 3), options)

    assert first.line_start == 1
    assert first.context_before == ()
    assert first.context_after == ("two\n", "", "", "", "")
    assert last.context_before == ("one\n",)
    assert last.context_after == ("", "", "", "", "")
    assert last.snippet == "two\n"


def test_enrich_end_char_between_cr_and_ln_stays_current_line(tmp_path: Path) -> None:
    # CRLF split across a chunk boundary: end_char points at the \n after the \r.
    # Must still resolve to line 1, normalize, and not be flagged stale.
    path = tmp_path / "split.txt"
    path.write_bytes(b"ab\r\ncd")
    result = SearchResult(
        label=0,
        score=1.0,
        file_path=path,
        chunk_text="ab",
        start_char=0,
        end_char=3,
        line_start=None,
        line_end=None,
        file_role=FileRole.source,
        language="text",
    )

    display = enrich_result(result, _options())

    assert display.stale_offsets is False
    assert display.line_start == 1
    assert display.line_end == 1
    assert display.snippet == "ab\n"
    assert display.context_after == ("cd",)
