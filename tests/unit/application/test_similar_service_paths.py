"""Unit contracts for SearchEngine's similar service paths and pure helpers.

Covers the unit-layer gaps left by the integration suite: the whole
`similar_path`/`similar_project` flows, open_session's own missing-vector-index
guard, the contrastive (--unlike) scoring branch, span-based self-exclusion
wiring, `_same_file`'s normpath fallback, and tokenize_query boundaries.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from simgrep.corpus import CorpusAccess
from simgrep.errors import SearchError
from simgrep.indexing import IndexEngine
from simgrep.models import SCHEMA_VERSION, Anchor, AppConfig, FreshnessMode, IndexOptions, ProjectConfig, SearchOptions, SimilarOptions
from simgrep.search import SearchEngine, _same_file, combine_candidate_scores, tokenize_query
from tests.conftest import FakeRuntime

COMMON = "def retry_request():\n    return retry(request)\n"


def _corpus(root: Path) -> Path:
    (root / "a.py").write_text(COMMON + "MARKER_A unique alpha\n", encoding="utf-8")
    (root / "b.py").write_text(COMMON + "MARKER_B unique beta\n", encoding="utf-8")
    (root / "c.py").write_text(COMMON + "MARKER_C unique gamma\n", encoding="utf-8")
    return root


def _project(root: Path) -> ProjectConfig:
    return ProjectConfig(SCHEMA_VERSION, "p", root, (root,), "fake", 128, 20)


def _anchor_options(root: Path, anchor_file: str, *, unlike_file: str | None = None, include_self: bool = False) -> SimilarOptions:
    text = (root / anchor_file).read_text(encoding="utf-8")
    return SimilarOptions(
        search=SearchOptions(query=text),
        anchor=Anchor(text=text, origin=(root / anchor_file).absolute(), start_char=0, end_char=len(text)),
        unlike=Anchor(text=(root / unlike_file).read_text(encoding="utf-8")) if unlike_file else None,
        include_self=include_self,
    )


class TestSimilarServicePaths:
    def test_similar_path_ephemeral_excludes_source_and_finds_duplicates(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        root = _corpus(tmp_path)
        outcome = SearchEngine(fake_runtime).similar_path(root, AppConfig(model="fake"), _anchor_options(root, "b.py"))

        names = [r.file_path.name for r in outcome.results]
        assert {"a.py", "c.py"} <= set(names)
        assert "b.py" not in names
        assert outcome.semantic_candidates > 0
        assert outcome.base_path == root

    def test_similar_project_skip_freshness_returns_duplicates(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        root = _corpus(tmp_path)
        project = _project(root)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

        outcome = SearchEngine(fake_runtime).similar_project(project, AppConfig(model="fake"), _anchor_options(root, "b.py"), FreshnessMode.skip)

        names = [r.file_path.name for r in outcome.results]
        assert {"a.py", "c.py"} <= set(names)
        assert "b.py" not in names
        assert outcome.files_seen == 3

    def test_open_project_raises_actionable_error_when_vector_index_missing_but_chunks_exist(
        self, tmp_path: Path, fake_runtime: FakeRuntime, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _corpus(tmp_path)
        project = _project(root)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))
        project.vector_index_path.unlink()
        access = CorpusAccess(fake_runtime)
        # Normally _ensure_ready raises first; neutralize it to prove the reader
        # open carries its own identical guard against TOCTOU races.
        monkeypatch.setattr(access, "_ensure_ready", lambda *args: None)

        def _open() -> None:
            with access.open_project(project, AppConfig(model="fake"), freshness=FreshnessMode.skip):
                pass  # pragma: no cover

        with pytest.raises(SearchError) as excinfo:
            _open()

        assert str(excinfo.value) == f"Vector index not found: {project.vector_index_path}"
        assert excinfo.value.hint == "Run `simgrep index --rebuild`."

    def test_contrastive_unlike_ephemeral_adds_unlike_contribution_and_demotes(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        root = _corpus(tmp_path)
        engine = SearchEngine(fake_runtime)
        app_config = AppConfig(model="fake")

        plain = engine.similar_path(root, app_config, _anchor_options(root, "b.py"))
        contrastive = engine.similar_path(root, app_config, _anchor_options(root, "b.py", unlike_file="a.py"))

        assert contrastive.results
        why = contrastive.results[0].why
        assert "semantic_like" in why and "semantic_unlike" in why
        assert max(r.score for r in plain.results) >= 0.9
        assert max(r.score for r in contrastive.results) < max(r.score for r in plain.results)

    @pytest.mark.parametrize(
        ("weight", "expected_label_1"),
        [(0.0, 3.0), (0.5, 2.0), (1.0, 1.0)],
    )
    def test_combine_candidate_scores_weight_boundaries(self, weight: float, expected_label_1: float) -> None:
        combined = dict(combine_candidate_scores({1: 3.0}, {1: 2.0, 2: 5.0}, weight))
        assert combined[1] == pytest.approx(expected_label_1)
        assert combined[2] == pytest.approx(-weight * 5.0)

    @pytest.mark.parametrize(("include_self", "expect_source"), [(False, False), (True, True)])
    def test_span_anchor_self_filtering_respects_include_self(self, tmp_path: Path, fake_runtime: FakeRuntime, include_self: bool, expect_source: bool) -> None:
        root = _corpus(tmp_path)
        outcome = SearchEngine(fake_runtime).similar_path(root, AppConfig(model="fake"), _anchor_options(root, "b.py", include_self=include_self))

        names = [r.file_path.name for r in outcome.results]
        assert ("b.py" in names) is expect_source

    def test_same_file_falls_back_to_normpath_when_samefile_raises_oserror(self, tmp_path: Path) -> None:
        ghost_a = tmp_path / "missing" / "x.py"
        ghost_b = tmp_path / "missing" / "." / "x.py"
        ghost_c = tmp_path / "missing" / "y.py"

        assert _same_file(ghost_a, ghost_b) is True
        assert _same_file(ghost_a, ghost_c) is False

    def test_tokenize_query_caps_output_at_eight_tokens(self) -> None:
        nine_plus = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        assert len(tokenize_query(nine_plus)) == 8
        assert tokenize_query(nine_plus)[-1] == "theta"
        eight = "alpha beta gamma delta epsilon zeta eta theta"
        assert tokenize_query(eight) == ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
        assert tokenize_query("HTTPServer") == ["http", "server"]
        assert tokenize_query("") == []

    @given(query=st.text(max_size=80))
    @settings(max_examples=150, deadline=None)
    def test_tokenize_query_properties_hold_for_arbitrary_text(self, query: str) -> None:
        tokens = tokenize_query(query)
        assert len(tokens) <= 8
        assert len(set(tokens)) == len(tokens)
        assert all(re.fullmatch(r"[a-z0-9]+", t) and len(t) > 1 for t in tokens)
