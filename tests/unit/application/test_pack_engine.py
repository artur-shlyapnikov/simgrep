"""Application-flow tests for the `simgrep pack` wiring in simgrep.main.

Fake SearchEngine (patched onto ``simgrep.search.SearchEngine``) drives the
per-query runs, label-dedup union, and pack_candidates composition without any
embedding machinery.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from simgrep.errors import SearchError
from simgrep.main import _run_pack
from simgrep.models import (
    AppConfig,
    FileRole,
    ScanOptions,
    SearchOptions,
    SearchOutcome,
    SearchResult,
)
from simgrep.pack import PackCandidate, PackOutcome

_ROLE = next(iter(FileRole))


def _result(label: int, score: float, path: str, text: str, *, line_start: int = 1) -> SearchResult:
    return SearchResult(
        label=label,
        score=score,
        file_path=Path(path),
        chunk_text=text,
        start_char=0,
        end_char=len(text),
        line_start=line_start,
        line_end=line_start + 3,
        file_role=_ROLE,
        language="py",
    )


class FakeSearchEngine:
    """Records per-query calls; serves canned results keyed by query text."""

    outcomes: dict[str, list[SearchResult]] = {}
    calls: list[tuple[str, int]] = []

    def __init__(self, runtime: object) -> None:
        del runtime

    def search_path(
        self,
        path: Path,
        app_config: AppConfig,
        options: SearchOptions,
        scan_options: ScanOptions | None = None,
    ) -> SearchOutcome:
        del app_config, scan_options
        FakeSearchEngine.calls.append((options.query, options.top))
        return SearchOutcome(
            results=list(FakeSearchEngine.outcomes.get(options.query, [])),
            base_path=path,
            files_seen=1,
            chunks_searched=len(FakeSearchEngine.outcomes.get(options.query, [])),
            semantic_candidates=0,
        )


@pytest.fixture(autouse=True)
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSearchEngine.outcomes = {}
    FakeSearchEngine.calls = []
    # _run_pack now routes through simgrep.execution.execute_pack, whose call-time
    # imports read these source-module attributes; patch them at their homes.
    monkeypatch.setattr("simgrep.search.SearchEngine", FakeSearchEngine, raising=False)
    monkeypatch.setattr("simgrep.config.load_app_config", lambda: AppConfig(), raising=False)


def _run(tmp_path: Path, queries: list[str], **kwargs: object) -> PackOutcome:
    defaults: dict[str, object] = {
        "budget": 3000,
        "lam": 0.7,
        "per_query": 8,
        "persistent": False,
        "ephemeral": True,
    }
    defaults.update(kwargs)
    return _run_pack(queries, tmp_path, **defaults)  # type: ignore[arg-type]


def test_union_dedups_by_label_keeping_max_score(tmp_path: Path) -> None:
    FakeSearchEngine.outcomes = {
        "q one": [_result(1, 0.90, "src/a.py", "alpha beta"), _result(2, 0.70, "src/b.py", "gamma")],
        "q two": [_result(1, 0.95, "src/a.py", "alpha beta"), _result(3, 0.60, "src/c.py", "delta")],
    }
    outcome = _run(tmp_path, ["q one", "q two"])
    assert isinstance(outcome, PackOutcome)
    assert outcome.pool_size == 3  # label 1 collapsed
    by_label = {sel.candidate.label: sel.candidate.score for sel in outcome.selections}
    assert by_label.get(1) == pytest.approx(0.95)  # max kept


def test_per_query_caps_top_of_each_run(tmp_path: Path) -> None:
    FakeSearchEngine.outcomes = {"q": [_result(1, 0.9, "a.py", "x")]}
    _run(tmp_path, ["q"], per_query=3)
    assert FakeSearchEngine.calls == [("q", 3)]


def test_flow_selects_under_budget_with_accounting(tmp_path: Path) -> None:
    FakeSearchEngine.outcomes = {
        "q": [
            _result(1, 0.90, "src/a.py", "alpha beta"),
            _result(2, 0.70, "src/b.py", "gamma delta"),
        ]
    }
    outcome = _run(tmp_path, ["q"], budget=1000)
    assert [sel.candidate.label for sel in outcome.selections] == [1, 2]
    assert all(not sel.truncated for sel in outcome.selections)
    assert outcome.dropped == 0
    assert outcome.used_tokens == sum(sel.tokens for sel in outcome.selections)
    assert outcome.used_tokens <= outcome.budget
    # Relative display paths anchored at the corpus root.
    assert outcome.selections[0].candidate.path == "src/a.py"


def test_candidates_are_pack_candidate_instances(tmp_path: Path) -> None:
    FakeSearchEngine.outcomes = {"q": [_result(7, 0.5, "n.md", "plain notes")]}
    outcome = _run(tmp_path, ["q"])
    sel = outcome.selections[0]
    assert isinstance(sel.candidate, PackCandidate)
    assert sel.candidate.label == 7
    assert sel.candidate.line_start == 1


def test_empty_results_yield_empty_pool(tmp_path: Path) -> None:
    outcome = _run(tmp_path, ["nothing", "matches"])
    assert outcome.selections == []
    assert outcome.pool_size == 0
    assert outcome.dropped == 0


def test_persistent_without_active_project_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("simgrep.project.find_active_project", lambda _path: None)
    with pytest.raises(SearchError):
        _run_pack(["q"], None, budget=1000, lam=0.7, per_query=8, persistent=True, ephemeral=False)


def test_line_fallback_when_result_lacks_lines(tmp_path: Path) -> None:
    result = SearchResult(
        label=5,
        score=0.4,
        file_path=Path("x.txt"),
        chunk_text="text",
        start_char=0,
        end_char=4,
        line_start=None,
        line_end=None,
        file_role=_ROLE,
        language="txt",
    )
    FakeSearchEngine.outcomes = {"q": [result]}
    outcome = _run(tmp_path, ["q"])
    assert outcome.selections[0].candidate.line_start == 1
