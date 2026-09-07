"""expand_results application tests: rebuild, memoization, graceful degrade."""

from __future__ import annotations

from pathlib import Path

import pytest

from simgrep.errors import ExpandError
from simgrep.expand import expand_results
from simgrep.models import FileRole, SearchResult

PY_FILE = "def alpha():\n    a = 1\n    b = 2\n"


def _result(path: Path, **overrides: object) -> SearchResult:
    defaults: dict[str, object] = {
        "label": 3,
        "score": 0.75,
        "file_path": path,
        "chunk_text": "b = 2",
        "start_char": PY_FILE.index("b = 2"),
        "end_char": PY_FILE.index("b = 2") + len("b = 2"),
        "line_start": 3,
        "line_end": 3,
        "file_role": FileRole.source,
        "language": "python",
        "why": {"kind": "semantic"},
    }
    defaults.update(overrides)
    return SearchResult(**defaults)  # type: ignore[arg-type]


class FakeFetch:
    def __init__(self, contents: dict[Path, str], fail: set[Path] | None = None) -> None:
        self.contents = contents
        self.fail = fail or set()
        self.calls: list[Path] = []

    def __call__(self, path: Path) -> str:
        self.calls.append(path)
        if path in self.fail:
            raise ExpandError(f"unreadable: {path}")
        return self.contents[path]


def test_rebuild_preserves_identity_fields(tmp_path: Path) -> None:
    src = tmp_path / "m.py"
    src.write_text(PY_FILE)
    fetch = FakeFetch({src: PY_FILE})
    original = _result(src)
    [expanded] = expand_results([original], fetch=fetch)
    assert expanded.label == original.label
    assert expanded.score == original.score
    assert expanded.file_path == original.file_path
    assert expanded.file_role == original.file_role
    assert expanded.language == original.language
    assert expanded.why == original.why


def test_expanded_bounds_and_text_cover_unit(tmp_path: Path) -> None:
    src = tmp_path / "m.py"
    src.write_text(PY_FILE)
    original = _result(src)
    [expanded] = expand_results([original], fetch=FakeFetch({src: PY_FILE}))
    assert expanded.chunk_text == "def alpha():\n    a = 1\n    b = 2"
    assert expanded.start_char == 0
    assert expanded.end_char == len("def alpha():\n    a = 1\n    b = 2")
    assert expanded.line_start == 1
    assert expanded.line_end == 3


def test_memoized_one_read_per_path(tmp_path: Path) -> None:
    src = tmp_path / "m.py"
    src.write_text(PY_FILE)
    other = tmp_path / "n.md"
    other_text = "para one\n\npara two\n"
    other.write_text(other_text)
    fetch = FakeFetch({src: PY_FILE, other: other_text})
    r1 = _result(src, label=1)
    r2 = _result(src, label=2, start_char=PY_FILE.index("a = 1"), chunk_text="a = 1")
    r3 = _result(other, start_char=other_text.index("para two"), chunk_text="para two")
    expanded = expand_results([r1, r2, r3], fetch=fetch)
    assert len(expanded) == 3
    assert fetch.calls.count(src) == 1
    assert fetch.calls.count(other) == 1


@pytest.mark.parametrize("why", [{"kind": "x"}, {}])
def test_empty_why_dict_preserved(why: dict[str, object], tmp_path: Path) -> None:
    src = tmp_path / "m.py"
    src.write_text(PY_FILE)
    original = _result(src, why=why)
    [expanded] = expand_results([original], fetch=FakeFetch({src: PY_FILE}))
    assert expanded.why == why


def test_unreadable_file_degrades_to_original(tmp_path: Path) -> None:
    src = tmp_path / "gone.py"
    original = _result(src)
    fetch = FakeFetch({}, fail={src})
    expanded = expand_results([original], fetch=fetch)
    assert expanded == [original]


def test_out_of_range_start_char_degrades_to_original(tmp_path: Path) -> None:
    src = tmp_path / "m.py"
    src.write_text(PY_FILE)
    stale = _result(src, start_char=10_000, end_char=11_000)
    expanded = expand_results([stale], fetch=FakeFetch({src: PY_FILE}))
    assert expanded == [stale]


def test_order_and_length_preserved_with_mixed_fates(tmp_path: Path) -> None:
    good = tmp_path / "good.py"
    good.write_text(PY_FILE)
    bad = tmp_path / "bad.py"
    ok1 = _result(good, label=1)
    degraded = _result(bad, label=2)
    ok2 = _result(good, label=3, start_char=PY_FILE.index("a = 1"), chunk_text="a = 1")
    fetch = FakeFetch({good: PY_FILE}, fail={bad})
    expanded = expand_results([ok1, degraded, ok2], fetch=fetch)
    assert [r.label for r in expanded] == [1, 2, 3]
    assert expanded[1] == degraded
    assert expanded[0].chunk_text.startswith("def alpha():")
    # both good hits expand to the same enclosing function unit starting at line 1
    assert expanded[2].line_start == 1
    assert expanded[2].chunk_text == "def alpha():\n    a = 1\n    b = 2"


def test_expand_never_truncates_large_units(tmp_path: Path) -> None:
    body = "".join(f"    x{i} = {i}\n" for i in range(500))
    big = f"def huge():\n{body}"
    src = tmp_path / "big.py"
    src.write_text(big)
    original = _result(src, start_char=big.index("x499"))
    [expanded] = expand_results([original], fetch=FakeFetch({src: big}))
    assert expanded.chunk_text == big[:-1]  # trailing newline excluded per pin
    assert (expanded.start_char, expanded.end_char) == (0, len(big) - 1)


def test_fetch_contract_is_injected_callable(tmp_path: Path) -> None:
    src = tmp_path / "m.py"
    src.write_text(PY_FILE)

    def fetch(path: Path) -> str:
        assert path == src
        return PY_FILE

    original = _result(src)
    assert expand_results([original], fetch=fetch)[0].start_char == 0


def test_fetch_oserror_still_degrades_gracefully(tmp_path: Path) -> None:
    """A fetch raising OSError (not ExpandError) must not crash expansion."""
    src = tmp_path / "m.py"

    def fetch(path: Path) -> str:
        raise OSError("permission denied")

    original = _result(src)
    assert expand_results([original], fetch=fetch) == [original]
