"""Engine-level integration tests for `simgrep similar`: persistent + ephemeral
flows, self-exclusion, and contrastive (--unlike) demotion with the fake runtime."""

from __future__ import annotations

from pathlib import Path

from simgrep.indexing import IndexEngine
from simgrep.models import (
    SCHEMA_VERSION,
    Anchor,
    AppConfig,
    FreshnessMode,
    IndexOptions,
    ProjectConfig,
    SearchOptions,
    SimilarOptions,
)
from simgrep.search import SearchEngine
from tests.conftest import FakeRuntime

COMMON = "def retry_request():\n    return retry(request)\n"


def _build_corpus(tmp_path: Path) -> Path:
    # Shared near-duplicate snippet first; distinct marker tail stays outside the
    # query-token window so sibling files pass the token-coverage gate.
    (tmp_path / "a.py").write_text(COMMON + "MARKER_A unique alpha\n", encoding="utf-8")
    (tmp_path / "b.py").write_text(COMMON + "MARKER_B unique beta\n", encoding="utf-8")
    (tmp_path / "c.py").write_text(COMMON + "MARKER_C unique gamma\n", encoding="utf-8")
    return tmp_path


def _project(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path,), "fake", 128, 20)


def _similar_options(tmp_path: Path, anchor_file: str, *, unlike: str | None = None, include_self: bool = False, top: int = 5) -> SimilarOptions:
    anchor_path = tmp_path / anchor_file
    text = anchor_path.read_text(encoding="utf-8")
    unlike_anchor = None
    if unlike is not None:
        unlike_anchor = Anchor(text=(tmp_path / unlike).read_text(encoding="utf-8"))
    return SimilarOptions(
        search=SearchOptions(query=text, top=top),
        anchor=Anchor(text=text, origin=anchor_path.absolute(), start_char=0, end_char=len(text)),
        unlike=unlike_anchor,
        include_self=include_self,
    )


class TestSimilarProject:
    def test_duplicates_surface_and_source_file_excluded(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        root = _build_corpus(tmp_path)
        project = _project(root)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

        outcome = SearchEngine(fake_runtime).similar_project(project, AppConfig(model="fake"), _similar_options(root, "a.py"), FreshnessMode.skip)

        names = [r.file_path.name for r in outcome.results]
        assert "b.py" in names and "c.py" in names, f"duplicates missing: {names}"
        assert "a.py" not in names, f"source file not excluded: {names}"

    def test_include_self_keeps_source_chunk(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        root = _build_corpus(tmp_path)
        project = _project(root)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

        engine = SearchEngine(fake_runtime)
        outcome = engine.similar_project(project, AppConfig(model="fake"), _similar_options(root, "a.py", include_self=True), FreshnessMode.skip)

        assert any(r.file_path.name == "a.py" for r in outcome.results)

    def test_contrastive_unlike_demotes_scores(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        root = _build_corpus(tmp_path)
        project = _project(root)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

        app_config = AppConfig(model="fake")
        plain = SearchEngine(fake_runtime).similar_project(project, app_config, _similar_options(root, "b.py"), FreshnessMode.skip)
        contrastive = SearchEngine(fake_runtime).similar_project(project, app_config, _similar_options(root, "b.py", unlike="a.py"), FreshnessMode.skip)

        assert contrastive.results, "contrastive run returned no results"
        assert max(r.score for r in contrastive.results) < max(
            r.score for r in plain.results
        ), "expected deterministic demotion when --unlike matches every candidate"

    def test_why_contains_both_contributions_with_unlike(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        root = _build_corpus(tmp_path)
        project = _project(root)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

        options = _similar_options(root, "b.py", unlike="a.py")
        outcome = SearchEngine(fake_runtime).similar_project(project, AppConfig(model="fake"), options, FreshnessMode.skip)

        assert outcome.results
        why = outcome.results[0].why
        assert "semantic_like" in why and "semantic_unlike" in why

    def test_why_shape_unchanged_without_unlike(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        root = _build_corpus(tmp_path)
        project = _project(root)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))
        search_outcome = SearchEngine(fake_runtime).search_project(
            project, AppConfig(model="fake"), SearchOptions(query="retry request retry_request"), FreshnessMode.skip
        )
        similar_outcome = SearchEngine(fake_runtime).similar_project(project, AppConfig(model="fake"), _similar_options(root, "b.py"), FreshnessMode.skip)
        assert similar_outcome.results
        assert set(similar_outcome.results[0].why) == set(search_outcome.results[0].why)


class TestSimilarEphemeral:
    def test_similar_path_ephemeral(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        root = _build_corpus(tmp_path)
        outcome = SearchEngine(fake_runtime).similar_path(root, AppConfig(model="fake"), _similar_options(root, "b.py"))
        names = [r.file_path.name for r in outcome.results]
        assert "c.py" in names
        assert "b.py" not in names


MARKERS = {
    "a.py": "MARKER_A unique alpha",
    "b.py": "MARKER_B unique beta",
    "c.py": "MARKER_C unique gamma",
}


def _anchor_options(root: Path, anchor_file: str, anchor_text: str, *, top: int = 5) -> SimilarOptions:
    anchor_path = root / anchor_file
    return SimilarOptions(
        search=SearchOptions(query=anchor_text, top=top),
        anchor=Anchor(text=anchor_text, origin=anchor_path.absolute(), start_char=0, end_char=len(anchor_text)),
    )


class TestSelfExclusionByteExactDecoding:
    """Anchors must decode like the indexer (utf-8-sig BOM strip, \r preserved)
    or stored chunk offsets drift and self-exclusion drops wrong spans."""

    def test_crlf_anchor_self_excluded(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        root = tmp_path
        for name, marker in MARKERS.items():
            (root / name).write_bytes((marker + "\r\n" + COMMON.replace("\n", "\r\n")).encode("utf-8"))
        project = _project(root)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))
        anchor_text = (root / "a.py").read_bytes().decode("utf-8")

        outcome = SearchEngine(fake_runtime).similar_project(project, AppConfig(model="fake"), _anchor_options(root, "a.py", anchor_text), FreshnessMode.skip)

        names = {r.file_path.name for r in outcome.results}
        assert "a.py" not in names
        assert {"b.py", "c.py"} <= names

    def test_bom_anchor_self_excluded(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        root = tmp_path
        for name, marker in MARKERS.items():
            body = marker + "\n" + COMMON
            data = (b"\xef\xbb\xbf" + body.encode("utf-8")) if name == "a.py" else body.encode("utf-8")
            (root / name).write_bytes(data)
        project = _project(root)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))
        anchor_text = (root / "a.py").read_bytes().decode("utf-8-sig")  # indexer strips the BOM

        outcome = SearchEngine(fake_runtime).similar_project(project, AppConfig(model="fake"), _anchor_options(root, "a.py", anchor_text), FreshnessMode.skip)

        names = {r.file_path.name for r in outcome.results}
        assert "a.py" not in names
        assert {"b.py", "c.py"} <= names


class TestSelfExclusionBeforeTruncation:
    def test_remaining_candidates_backfill_to_top(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        root = tmp_path
        markers = dict(MARKERS)
        markers["d.py"] = "MARKER_D unique delta"
        markers["e.py"] = "MARKER_E unique epsilon"
        for name, marker in markers.items():
            (root / name).write_text(marker + "\n" + COMMON, encoding="utf-8")
        project = _project(root)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

        outcome = SearchEngine(fake_runtime).similar_project(
            project, AppConfig(model="fake"), _anchor_options(root, "a.py", markers["a.py"] + "\n" + COMMON, top=3), FreshnessMode.skip
        )

        names = [r.file_path.name for r in outcome.results]
        assert len(outcome.results) == 3, f"expected backfill to --top, got {names}"
        assert "a.py" not in names
