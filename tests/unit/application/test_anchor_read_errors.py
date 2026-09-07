from __future__ import annotations

from pathlib import Path

import pytest

from simgrep import search
from simgrep.errors import SearchError


def test_unreadable_anchor_raises_search_error(tmp_path: Path) -> None:
    path = tmp_path / "anchor.txt"
    path.write_bytes(b"secret")
    path.chmod(0o000)

    with pytest.raises(SearchError, match=str(path)):
        search._read_anchor_text(path)


def test_missing_anchor_raises_search_error(tmp_path: Path) -> None:
    path = tmp_path / "missing.txt"

    with pytest.raises(SearchError):
        search._read_anchor_text(path)


def test_non_utf8_anchor_raises_search_error(tmp_path: Path) -> None:
    path = tmp_path / "binary.bin"
    path.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(SearchError, match="not valid UTF-8"):
        search._read_anchor_text(path)
