"""Regression tests for _diversify mode semantics.

Pins the documented result-diversity contract:
- window: drop a row whose file already appears in the preceding 2 results;
- file: keep at most one result per file.
Package-mode capping is pinned by test_ranking_refactor_slice.py and left alone here.
"""

from pathlib import Path

from simgrep.corpus import StoredChunk
from simgrep.models import DiversityMode, LexicalFallbackMode, SearchOptions
from simgrep.ranking import rank_candidates
from tests.conftest import _ranking_chunk


def _options(top: int, diversity: DiversityMode) -> SearchOptions:
    return SearchOptions(
        query="x",
        top=top,
        lexical_weight=0.3,
        diversity=diversity,
        path_boosts=(),
        lexical_fallback=LexicalFallbackMode.fill,
        min_score=0.0,
    )


def _rank(files: tuple[str, ...], top: int, diversity: DiversityMode) -> list[Path]:
    matches = [(i, 1.0 - i * 0.01) for i in range(1, len(files) + 1)]
    rows: list[StoredChunk] = [_ranking_chunk(i, name, f"chunk {i}", "source") for i, name in enumerate(files, start=1)]
    ranked = rank_candidates(
        query="x",
        semantic_matches=matches,
        semantic_rows=rows,
        lexical_rows=[],
        options=_options(top, diversity),
    )
    return [r.file_path for r in ranked]


class TestWindowMode:
    def test_suppresses_adjacent_run_of_same_file(self) -> None:
        files = _rank(("a.py", "a.py", "b.py", "a.py", "a.py"), top=5, diversity=DiversityMode.window)
        assert files == [Path("a.py"), Path("b.py")]

    def test_allows_reentry_after_two_other_files(self) -> None:
        files = _rank(
            ("a.py", "b.py", "c.py", "a.py", "d.py", "e.py", "a.py"),
            top=7,
            diversity=DiversityMode.window,
        )
        assert files == [
            Path("a.py"),
            Path("b.py"),
            Path("c.py"),
            Path("a.py"),
            Path("d.py"),
            Path("e.py"),
            Path("a.py"),
        ]


class TestFileMode:
    def test_keeps_one_result_per_file(self) -> None:
        files = _rank(("a.py", "a.py", "b.py", "a.py"), top=4, diversity=DiversityMode.file)
        assert files == [Path("a.py"), Path("b.py")]
