from simgrep.text import compute_line_starts, expand_offsets_to_line_bounds, offset_to_line


def test_line_offsets_helpers() -> None:
    text = "a\nbb\nccc"
    starts = compute_line_starts(text)
    assert starts == [0, 2, 5]
    assert offset_to_line(starts, 0) == 1
    assert offset_to_line(starts, 3) == 2
    start, end, expanded_start, expanded_end = expand_offsets_to_line_bounds(text, 3, 4)
    assert (start, end) == (3, 4)
    assert (expanded_start, expanded_end) == (2, 5)
    assert expand_offsets_to_line_bounds("abc\ndef", 2, 2) == (2, 2, 2, 2)
    assert expand_offsets_to_line_bounds("abc", 3, 1) == (3, 1, 3, 1)


def test_expand_final_line_without_trailing_newline_single_line_chunk() -> None:
    start, end, expanded_start, expanded_end = expand_offsets_to_line_bounds("abc\ndefxyz", 6, 9)
    assert (start, end) == (6, 9)
    # Chunk on the FINAL unterminated line passes through unexpanded:
    # commit 0940064 — expanding every sliding window on single-line texts
    # collapsed them into duplicate whole-text chunks.
    assert (expanded_start, expanded_end) == (6, 9)


def test_expand_first_line_fully_covered_extends_over_newline_offset_only() -> None:
    start, end, expanded_start, expanded_end = expand_offsets_to_line_bounds("abc\ndefxyz", 0, 3)
    assert (start, end) == (0, 3)
    assert (expanded_start, expanded_end) == (0, 4)


def test_expand_multi_line_final_chunk_expands_to_text_end() -> None:
    _, _, _, expanded_end = expand_offsets_to_line_bounds("abc\ndefxyz", 0, 6)
    assert expanded_end == 10


def test_expand_newline_terminated_file_unchanged() -> None:
    _, _, _, expanded_end = expand_offsets_to_line_bounds("abc\ndef\n", 4, 7)
    assert expanded_end == 8


def test_expand_max_extra_chars_guard_still_respected() -> None:
    huge_line = "x" * 1500
    text = "short\n" + huge_line
    chunk_start, chunk_end = len(text) - 3, len(text)
    start, end, expanded_start, expanded_end = expand_offsets_to_line_bounds(text, chunk_start, chunk_end)
    assert (start, end) == (chunk_start, chunk_end)
    assert (expanded_start, expanded_end) == (chunk_start, chunk_end)
