"""Lexical semantic-unit expansion (``simgrep expand`` / ``--whole-unit``).

Pure string math over raw file text — no AST, no new dependencies, deterministic.
``unit_bounds`` per-family algorithms are pinned by design; all tie-breaks resolve
by ascending line index.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from simgrep.errors import ExpandError
from simgrep.models import SearchResult
from simgrep.text import compute_line_starts, offset_to_line

__all__ = [
    "cap_unit",
    "expand_results",
    "read_text_raw",
    "unit_bounds",
    "unit_family",
]

_DEDENT_SUFFIXES = {".py", ".pyw", ".yaml", ".yml", ".haml", ".sass", ".styl"}
_BRACE_SUFFIXES = {
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".hpp",
    ".cs",
    ".java",
    ".kt",
    ".kts",
    ".swift",
    ".rs",
    ".go",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".json",
    ".php",
}

_BLOCK_KEYWORDS = (
    "def",
    "class",
    "if",
    "elif",
    "else",
    "for",
    "while",
    "try",
    "except",
    "finally",
    "with",
    "match",
    "case",
)

_NAMED_OPENER_RE = re.compile(r"\b(func|fn|function|class|struct|impl|interface|enum|namespace|constructor|def)\b")


def unit_family(path: Path) -> str:
    """Map a path to its expansion family: ``"dedent" | "brace" | "paragraph"``."""
    suffix = path.suffix.lower()
    if suffix in _DEDENT_SUFFIXES:
        return "dedent"
    if suffix in _BRACE_SUFFIXES:
        return "brace"
    return "paragraph"


def read_text_raw(path: Path) -> str:
    """Read file bytes without newline translation (stored offsets refer to RAW text).

    utf-8 first, latin-1 fallback. Raises :class:`ExpandError` on OSError.
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ExpandError(f"cannot read {path}: {exc}", hint="check that the path exists and is readable") from exc
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


@dataclass(frozen=True)
class _LineSpans:
    """Per-line offsets of a text split strictly on ``\\n``."""

    starts: list[int]
    ends: list[int]  # end of line CONTENT (excludes trailing \n / \r\n)
    blanks: list[bool]  # True when the line has no visible content

    @property
    def count(self) -> int:
        return len(self.starts)

    def text(self, source: str, i: int) -> str:
        """Content of line ``i`` without its terminator."""
        return source[self.starts[i] : self.ends[i]]

    def blank(self, i: int) -> bool:
        return self.blanks[i]


def _line_spans(text: str) -> _LineSpans:
    starts: list[int] = []
    ends: list[int] = []
    pos = 0
    while True:
        nl = text.find("\n", pos)
        if nl == -1:
            starts.append(pos)
            ends.append(len(text))
            break
        content_end = nl - 1 if nl > pos and text[nl - 1] == "\r" else nl
        starts.append(pos)
        ends.append(content_end)
        pos = nl + 1
        if pos == len(text):  # trailing newline: drop the phantom empty last line
            break
    blanks = [not text[s:e].strip() for s, e in zip(starts, ends)]
    return _LineSpans(starts=starts, ends=ends, blanks=blanks)


def _line_of(starts: list[int], offset: int) -> int:
    return bisect_right(starts, offset) - 1


def _indent(line: str) -> int:
    width = 0
    for ch in line:
        if ch == " ":
            width += 1
        elif ch == "\t":
            width += 4
        else:
            break
    return width


def _is_block_starter(stripped: str) -> bool:
    return stripped.endswith(":") or stripped.startswith(_BLOCK_KEYWORDS)


def unit_bounds(text: str, offset: int, *, family: str = "paragraph") -> tuple[int, int]:
    """Bounds ``(start_char, end_char)`` — end exclusive — of the unit containing ``offset``.

    ``family`` is one of :func:`unit_family`'s outputs; callers deriving it from a
    path pass ``unit_family(path)`` (a CLI ``--language`` override maps here too).
    Raises ValueError when ``text`` is empty or ``offset`` is out of range.
    """
    if not text:
        raise ValueError("cannot expand empty text")
    if not 0 <= offset < len(text):
        raise ValueError(f"offset {offset} out of range for text of length {len(text)}")
    spans = _line_spans(text)
    hit = _line_of(spans.starts, offset)
    if family == "dedent":
        return _dedent_bounds(text, spans, hit)
    if family == "brace":
        return _brace_bounds(text, spans, hit, offset)
    return _paragraph_bounds(text, spans, hit)


def _paragraph_bounds(text: str, spans: _LineSpans, hit: int) -> tuple[int, int]:
    """Contiguous run of non-blank lines containing ``hit``; blank hits expand below."""
    n = spans.count
    start_line = hit
    if spans.blank(hit):
        j = hit
        while j < n and spans.blank(j):
            j += 1
        if j == n:  # nothing but blanks below: take the block above instead
            k = hit
            while k > 0 and spans.blank(k - 1):
                k -= 1
            start_line = max(k - 1, 0)
        else:
            start_line = j
    s = start_line
    while s > 0 and not spans.blank(s - 1):
        s -= 1
    e = start_line
    while e + 1 < n and not spans.blank(e + 1):
        e += 1
    start_char, end_char = spans.starts[s], spans.ends[e]
    # A unit is never empty: a blank-only neighborhood collapses to the full text.
    if end_char <= start_char or not text[start_char:end_char].strip():
        return 0, len(text)
    return start_char, end_char


def _dedent_bounds(text: str, spans: _LineSpans, hit: int) -> tuple[int, int]:
    n = spans.count
    anchor = hit
    while anchor >= 0 and spans.blank(anchor):
        anchor -= 1
    if anchor < 0:  # only blank lines above: fall forward to the next real line
        fwd = hit
        while fwd < n and spans.blank(fwd):
            fwd += 1
        if fwd == n:
            return 0, len(text)
        anchor = fwd

    header: int | None = None
    if _is_block_starter(spans.text(text, anchor).strip()):
        header = anchor  # hit ON a header line: its own unit
    else:
        anchor_indent = _indent(spans.text(text, anchor))
        for i in range(anchor, -1, -1):
            stripped = spans.text(text, i).strip()
            if stripped and _indent(spans.text(text, i)) < anchor_indent and _is_block_starter(stripped):
                header = i
                break
    if header is None:
        return _paragraph_bounds(text, spans, anchor)

    header_indent = _indent(spans.text(text, header))
    end_line = header
    j = header + 1
    while j < n:
        if spans.blank(j):
            j += 1
            continue
        if _indent(spans.text(text, j)) <= header_indent:
            break
        end_line = j
        j += 1
    return spans.starts[header], spans.ends[end_line]


@dataclass(frozen=True)
class _BraceEvent:
    pos: int
    depth: int  # opens: depth AFTER '{'; closes: depth BEFORE '}'


def _scan_braces(text: str) -> tuple[list[_BraceEvent], list[_BraceEvent]]:
    """Single lexical scan: structural brace events outside strings/comments.

    Documented approximations: raw braces inside string/char literals and
    ``//`` ``#`` ``/* */`` comments are ignored; template literals and regex
    literals are not modelled.
    """
    opens: list[_BraceEvent] = []
    closes: list[_BraceEvent] = []
    depth = 0
    i = 0
    n = len(text)
    normal, line_comment, block_comment, string = range(4)
    state = normal
    quote = ""
    while i < n:
        ch = text[i]
        if state == normal:
            if ch == "/" and i + 1 < n and text[i + 1] == "/":
                state = line_comment
                i += 2
            elif ch == "#":
                state = line_comment
                i += 1
            elif ch == "/" and i + 1 < n and text[i + 1] == "*":
                state = block_comment
                i += 2
            elif ch in "\"'":
                state = string
                quote = ch
                i += 1
            elif ch == "{":
                depth += 1
                opens.append(_BraceEvent(i, depth))
                i += 1
            elif ch == "}":
                if depth > 0:
                    closes.append(_BraceEvent(i, depth))
                    depth -= 1
                i += 1
            else:
                i += 1
        elif state == line_comment:
            if ch == "\n":
                state = normal
            i += 1
        elif state == block_comment:
            if ch == "*" and i + 1 < n and text[i + 1] == "/":
                state = normal
                i += 2
            else:
                i += 1
        else:  # string
            if ch == "\\":
                i += 2
            elif ch == quote:
                state = normal
                i += 1
            else:
                i += 1
    return opens, closes


def _looks_like_definition(text: str, spans: _LineSpans, ev: _BraceEvent) -> bool:
    """PINNED rule: '(' before the '{' on its line, or a known definition keyword."""
    line_idx = _line_of(spans.starts, ev.pos)
    before = text[spans.starts[line_idx] : ev.pos]
    return "(" in before or bool(_NAMED_OPENER_RE.search(spans.text(text, line_idx).strip()))


def _brace_bounds(text: str, spans: _LineSpans, hit: int, offset: int) -> tuple[int, int]:
    opens, closes = _scan_braces(text)
    open_pos = [ev.pos for ev in opens]
    close_pos = [ev.pos for ev in closes]
    depth_at_hit = bisect_left(open_pos, offset) - bisect_left(close_pos, offset)
    if depth_at_hit <= 0:
        return _paragraph_bounds(text, spans, hit)  # no enclosing region (incl. unbalanced files)

    # Innermost-first chain of enclosing opener events.
    chain: list[_BraceEvent] = []
    limit = offset
    needed = depth_at_hit
    for ev in reversed(opens):
        if needed == 0:
            break
        if ev.depth == needed and ev.pos < limit:
            chain.append(ev)
            limit = ev.pos
            needed -= 1
    if needed > 0:
        return _paragraph_bounds(text, spans, hit)  # unbalanced/truncated

    chosen = chain[0]
    for ev in chain:  # prefer a definition-looking opener
        if _looks_like_definition(text, spans, ev):
            chosen = ev
            break
    close = next((c for c in closes if c.depth == chosen.depth and c.pos >= offset), None)
    if close is None:
        return _paragraph_bounds(text, spans, hit)  # region never closes: truncated file
    start_line = _line_of(spans.starts, chosen.pos)
    if chosen.pos == chain[0].pos:
        # No named opener anywhere in the chain (e.g. Allman-style '{' on its own
        # line): extend upward across contiguous signature/header lines. Stop at
        # blank lines or lines ending with ';', '{', '}'.
        j = start_line
        while j > 0:
            prev = spans.text(text, j - 1).strip()
            if not prev or prev.endswith((";", "{", "}")):
                break
            j -= 1
        start_line = j
    return spans.starts[start_line], close.pos + 1


def cap_unit(text: str, start: int, end: int, *, max_chars: int, anchor: int) -> tuple[int, int]:
    """Clamp ``[start, end)`` to a window of ``max_chars`` chars containing ``anchor``.

    Prefers the head window ``[start, start+max_chars)`` when the anchor fits;
    otherwise centers on the anchor. Window edges snap outward to line boundaries,
    never leaving ``[start, end)``.
    """
    if end - start <= max_chars:
        return start, end
    if anchor - start < max_chars:
        win_start = start
        win_end = start + max_chars
    else:
        win_start = max(start, min(anchor - max_chars // 2, end - max_chars))
        win_end = min(win_start + max_chars, end)
    starts = compute_line_starts(text)
    win_start = starts[bisect_right(starts, win_start) - 1]
    end_line = bisect_right(starts, win_end - 1)
    # Snap the tail to the next line start, but never past the unit end and
    # never past the char budget when the window already ends at EOF.
    win_end = min(starts[end_line], end) if end_line < len(starts) else min(win_end, end)
    return win_start, win_end


def expand_results(results: list[SearchResult], *, fetch: Callable[[Path], str]) -> list[SearchResult]:
    """Expand every result to its enclosing semantic unit.

    ``fetch`` reads raw text (raising :class:`ExpandError` on unreadable files);
    it is called at most once per distinct path. Unreadable files and stale
    out-of-range offsets degrade gracefully to the ORIGINAL result — this function
    never raises. Order and length are preserved. No truncation happens here;
    callers apply ``cap_unit`` / render-level ``max_chars`` afterwards.
    """
    cache: dict[Path, str | None] = {}
    out: list[SearchResult] = []
    for result in results:
        if result.file_path not in cache:
            try:
                cache[result.file_path] = fetch(result.file_path)
            except Exception:  # noqa: BLE001 - graceful-degrade contract
                cache[result.file_path] = None
        text = cache[result.file_path]
        if text is None:
            out.append(result)
            continue
        try:
            bounds = unit_bounds(text, result.start_char, family=unit_family(result.file_path))
        except ValueError:
            out.append(result)
            continue
        start, end = bounds
        line_starts = compute_line_starts(text)
        out.append(
            SearchResult(
                label=result.label,
                score=result.score,
                file_path=result.file_path,
                chunk_text=text[start:end],
                start_char=start,
                end_char=end,
                line_start=offset_to_line(line_starts, start),
                line_end=offset_to_line(line_starts, max(start, end - 1)),
                file_role=result.file_role,
                language=result.language,
                why=result.why,
            )
        )
    return out
