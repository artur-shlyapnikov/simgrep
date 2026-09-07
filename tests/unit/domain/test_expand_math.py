"""Pure-math tests for simgrep.expand: unit_bounds, cap_unit, unit_family."""

from __future__ import annotations

from pathlib import Path

import pytest

from simgrep.expand import cap_unit, unit_bounds, unit_family


def _bounds_of(text: str, needle: str, family: str) -> tuple[int, int]:
    """Offset of `needle` inside `text` -> unit_bounds result."""
    offset = text.index(needle) + len(needle) // 2
    return unit_bounds(text, offset, family=family)


# ---------------------------------------------------------------- dedent family

PY_FUNC = "def outer():\n    a = 1\n    b = 2\n\n\ndef other():\n    pass\n"


def test_dedent_simple_function() -> None:
    start, end = _bounds_of(PY_FUNC, "b = 2", "dedent")
    assert PY_FUNC[start:end] == "def outer():\n    a = 1\n    b = 2"


def test_dedent_hit_on_header_line() -> None:
    start, end = _bounds_of(PY_FUNC, "def outer", "dedent")
    assert PY_FUNC[start:end] == "def outer():\n    a = 1\n    b = 2"


def test_dedent_nested_function_innermost_wins() -> None:
    text = "def outer():\n" "    x = 1\n" "\n" "    def inner():\n" "        y = 2\n" "        return y\n" "\n" "    return x\n"
    start, end = _bounds_of(text, "y = 2", "dedent")
    assert text[start:end] == "    def inner():\n        y = 2\n        return y"


def test_dedent_class_method() -> None:
    text = "class Repo:\n" "    attr = 0\n" "\n" "    def load(self):\n" "        data = 1\n" "        return data\n"
    # hit in method body -> nearest lower-indent block starter is the def
    start, end = _bounds_of(text, "data = 1", "dedent")
    assert text[start:end] == "    def load(self):\n        data = 1\n        return data"
    # hit on class attribute -> the whole class body
    start, end = _bounds_of(text, "attr = 0", "dedent")
    assert text[start:end] == text[:-1]


def test_dedent_decorator_excluded_from_unit() -> None:
    text = "@decorator\ndef run():\n    work()\n"
    start, end = _bounds_of(text, "work()", "dedent")
    assert text[start:end] == "def run():\n    work()"


def test_dedent_tabs_count_as_four() -> None:
    text = "def t():\n\tif True:\n\t\treturn 1\n"
    start, end = _bounds_of(text, "return 1", "dedent")
    assert text[start:end] == "\tif True:\n\t\treturn 1"


def test_dedent_no_trailing_newline_at_eof() -> None:
    text = "def tail():\n    last = 1"  # no trailing newline
    start, end = _bounds_of(text, "last = 1", "dedent")
    assert text[start:end] == "def tail():\n    last = 1"
    assert end == len(text)


def test_dedent_one_liner_def_is_single_line_unit() -> None:
    text = "def f(): return 1\n\nnext_thing = 2\n"
    start, end = _bounds_of(text, "return 1", "dedent")
    assert text[start:end] == "def f(): return 1"


def test_dedent_no_block_starter_falls_back_to_paragraph() -> None:
    text = "alpha = 1\nbeta = 2\n\ngamma = 3\n"
    start, end = _bounds_of(text, "beta = 2", "dedent")
    assert text[start:end] == "alpha = 1\nbeta = 2"


def test_dedent_trailing_blank_lines_trimmed() -> None:
    text = "def f():\n    a = 1\n    b = 2\n\n\nafter()\n"
    start, end = _bounds_of(text, "b = 2", "dedent")
    assert text[start:end] == "def f():\n    a = 1\n    b = 2"


def test_dedent_yaml_style_blocks() -> None:
    text = "services:\n  web:\n    image: nginx\n  db:\n    image: pg\n"
    start, end = _bounds_of(text, "image: pg", "dedent")
    assert text[start:end] == "  db:\n    image: pg"


# ---------------------------------------------------------------- brace family

C_FUNC = "int main(void)\n{\n    int x = 1;\n    return x;\n}\n"


def test_brace_c_function() -> None:
    start, end = _bounds_of(C_FUNC, "int x = 1;", "brace")
    assert C_FUNC[start:end] == C_FUNC[: C_FUNC.rindex("}") + 1]
    assert end == C_FUNC.index("}", start) + 1


def test_brace_conditional_block_counts_as_definition_opener() -> None:
    # PINNED literal rule: an opener line containing '(' before '{' is definition-like.
    text = "void handler(int n)\n" "{\n" "    if (n > 0)\n" "    {\n" "        log(n);\n" "    }\n" "}\n"
    start, end = _bounds_of(text, "log(n);", "brace")
    assert text[start:end] == "    if (n > 0)\n    {\n        log(n);\n    }"


def test_brace_prefers_named_encloser_over_bare_block() -> None:
    text = "void f(void) {\n" "    {\n" "        deep();\n" "    }\n" "}\n"
    start, end = _bounds_of(text, "deep();", "brace")
    assert text[start:end] == text[: text.rindex("}") + 1]


def test_brace_string_braces_ignored() -> None:
    text = 'void f(void)\n{\n    char *s = "}{";\n    use(s);\n}\n'
    start, end = _bounds_of(text, "use(s);", "brace")
    assert text[start:end] == text[: text.rindex("}") + 1]


def test_brace_comment_braces_ignored() -> None:
    text = "void f(void)\n{\n    /* } { */\n    // }\n    run();\n}\n"
    start, end = _bounds_of(text, "run();", "brace")
    assert text[start:end] == text[: text.rindex("}") + 1]


def test_brace_unbalanced_falls_back_to_paragraph() -> None:
    text = "void f(void)\n{\n    truncated(1;\n"  # missing closing brace
    start, end = _bounds_of(text, "truncated(1;", "brace")
    # paragraph fallback: contiguous non-blank lines containing the hit = whole file here
    assert text[start:end] == text[:-1]


def test_brace_opener_same_line_as_brace() -> None:
    text = "int go(void) {\n    step();\n}\n"
    start, end = _bounds_of(text, "step();", "brace")
    assert text[start:end] == "int go(void) {\n    step();\n}"


# ---------------------------------------------------------------- paragraph family


def test_paragraph_prose_block() -> None:
    text = "First para line one.\nFirst para line two.\n\nSecond para.\n"
    start, end = _bounds_of(text, "line two", "paragraph")
    assert text[start:end] == "First para line one.\nFirst para line two."


def test_paragraph_bof_and_eof_edges() -> None:
    text = "only one block\nwith two lines"
    start, end = _bounds_of(text, "two lines", "paragraph")
    assert (start, end) == (0, len(text))


def test_paragraph_blank_hit_line_expands_below() -> None:
    text = "para one\n\n   \npara two\n"
    offset = text.index("   \n") + 1
    start, end = unit_bounds(text, offset, family="paragraph")
    assert text[start:end] == "para two"


def test_paragraph_all_blank_file_never_yields_empty_unit() -> None:
    text = "\n   \n\n"
    start, end = unit_bounds(text, 0, family="paragraph")
    assert (start, end) == (0, len(text))
    assert end > start


def test_paragraph_blank_only_tail_file_never_yields_empty_unit() -> None:
    text = "word\n\n\n\n"
    offset = len(text) - 1
    start, end = unit_bounds(text, offset, family="paragraph")
    # Nothing but blanks below the hit: falls back to the block above, never empty.
    assert text[start:end] == "word"


def test_paragraph_crlf_preserves_raw_newlines_and_excludes_trailing_cr() -> None:
    text = "first\r\nsecond\r\nthird\r\n"
    offset = text.index("second")
    start, end = unit_bounds(text, offset, family="paragraph")
    assert text[start:end] == "first\r\nsecond\r\nthird"


# ---------------------------------------------------------------- cap_unit

MULTILINE = "".join(f"line{i} = {i}\n" for i in range(20))


def test_cap_unit_fits_unchanged() -> None:
    assert cap_unit(MULTILINE, 0, len(MULTILINE), max_chars=10_000, anchor=5) == (0, len(MULTILINE))


def test_cap_unit_head_window_when_anchor_near_start() -> None:
    start, end = cap_unit(MULTILINE, 0, len(MULTILINE), max_chars=20, anchor=3)
    assert (start, end) == (0, 20)


def test_cap_unit_anchor_centered() -> None:
    anchor = MULTILINE.index("line15")
    start, end = cap_unit(MULTILINE, 0, len(MULTILINE), max_chars=40, anchor=anchor)
    assert start <= anchor < end
    assert end - start <= 40 + len("line19 = 19\n")  # snap slack only


def test_cap_unit_snaps_to_line_boundaries() -> None:
    anchor = MULTILINE.index("line10")
    start, end = cap_unit(MULTILINE, 0, len(MULTILINE), max_chars=25, anchor=anchor)
    assert MULTILINE[start : start + 4].startswith("line")
    assert end == len(MULTILINE) or MULTILINE[end - 1] == "\n"


def test_cap_unit_window_never_leaves_original_bounds() -> None:
    anchor = len(MULTILINE) - 5
    start, end = cap_unit(MULTILINE, 7, len(MULTILINE), max_chars=30, anchor=anchor)
    assert start >= 7 and end <= len(MULTILINE)
    assert start <= anchor < end


def test_cap_unit_single_line_file_respects_budget() -> None:
    text = "x" * 506
    assert cap_unit(text, 0, len(text), max_chars=100, anchor=0) == (0, 100)


def test_cap_unit_tail_snap_never_leaves_original_bounds() -> None:
    text = "aaa\n" + "b" * 300 + "\ntail\n"
    unit_start, unit_end = 4, 304  # the long middle line only
    start, end = cap_unit(text, unit_start, unit_end, max_chars=100, anchor=200)
    assert unit_start <= start < end <= unit_end


# ---------------------------------------------------------------- family mapping


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        (".py", "dedent"),
        (".pyw", "dedent"),
        (".yaml", "dedent"),
        (".yml", "dedent"),
        (".rs", "brace"),
        (".tsx", "brace"),
        (".json", "brace"),
        (".go", "brace"),
        (".md", "paragraph"),
        (".sh", "paragraph"),
        (".weird", "paragraph"),
    ],
)
def test_unit_family_mapping(suffix: str, expected: str) -> None:
    assert unit_family(Path(f"/tmp/some/file{suffix}")) == expected


def test_unit_family_uppercase_suffix_normalized() -> None:
    assert unit_family(Path("/tmp/FILE.PY")) == "dedent"


# ---------------------------------------------------------------- offset edges


def test_offset_zero_is_valid() -> None:
    text = "def a():\n    pass\n"
    start, end = unit_bounds(text, 0, family="paragraph")
    assert text[start:end] == "def a():\n    pass"


def test_offset_at_last_char_is_valid() -> None:
    text = "x = 1\n"
    start, end = unit_bounds(text, len(text) - 1, family="paragraph")
    assert text[start:end] == "x = 1"


@pytest.mark.parametrize("offset", [-1, len(PY_FUNC), len(PY_FUNC) + 100])
def test_offset_out_of_range_raises_value_error(offset: int) -> None:
    with pytest.raises(ValueError):
        unit_bounds(PY_FUNC, offset, family="dedent")


def test_empty_text_raises_value_error() -> None:
    with pytest.raises(ValueError):
        unit_bounds("", 0)
