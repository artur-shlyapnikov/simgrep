from __future__ import annotations

from bisect import bisect_right


def compute_line_starts(text: str) -> list[int]:
    starts = [0]
    pos = text.find("\n")
    while pos != -1:
        starts.append(pos + 1)
        pos = text.find("\n", pos + 1)
    return starts


def offset_to_line(line_starts: list[int], offset: int) -> int:
    return bisect_right(line_starts, max(0, offset))


def expand_offsets_to_line_bounds(
    text: str,
    start: int,
    end: int,
    *,
    max_extra_chars: int = 1000,
    line_starts: list[int] | None = None,
) -> tuple[int, int, int, int]:
    if start >= end:
        return start, end, start, end

    starts = line_starts if line_starts is not None else compute_line_starts(text)
    line_start = offset_to_line(starts, start)
    line_end = offset_to_line(starts, end - 1)
    expanded_start = starts[line_start - 1] if line_start > 0 else 0
    if line_end < len(starts):
        expanded_end = starts[line_end]
    elif line_start == line_end:
        return start, end, start, end
    else:
        expanded_end = len(text)
    if expanded_end - expanded_start - (end - start) > max_extra_chars:
        return start, end, start, end
    return start, end, expanded_start, expanded_end
