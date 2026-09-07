from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simgrep.corpus import CorpusAccess
from simgrep.errors import SearchError
from simgrep.indexing import IndexEngine
from simgrep.models import SCHEMA_VERSION, AppConfig, FreshnessMode, IndexOptions, ProjectConfig, SearchOptions
from simgrep.search import SearchEngine, effective_candidate_top, tokenize_query
from simgrep.store import Store
from tests.conftest import FakeRuntime, FakeVectorIndex


def _project(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path,), "fake", 128, 20)


def test_tokenize_query() -> None:
    assert tokenize_query("rollbackPayment now") == ["rollback", "payment", "now"]
    assert tokenize_query("") == []
    assert tokenize_query("one two three four five six seven eight nine ten") == [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
    ]


def test_search_project_check_raises_on_stale(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    project = _project(tmp_path)
    (tmp_path / "a.py").write_text("before", encoding="utf-8")
    app_config = AppConfig(model="fake")
    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True))
    (tmp_path / "b.py").write_text("after", encoding="utf-8")
    with pytest.raises(SearchError, match="stale"):
        SearchEngine(fake_runtime).search_project(project, app_config, SearchOptions(query="after"), FreshnessMode.check)


def test_search_path_ephemeral_returns_results(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    (tmp_path / "a.py").write_text("rollback payment", encoding="utf-8")
    outcome = SearchEngine(fake_runtime).search_path(tmp_path, AppConfig(model="fake"), SearchOptions(query="rollback", lexical_top=0, lexical_weight=0.0))
    assert outcome.results
    assert outcome.results[0].file_path.name == "a.py"


def test_search_project_skip_rejects_failed_state(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    project = _project(tmp_path)
    (tmp_path / "a.py").write_text("before", encoding="utf-8")
    app_config = AppConfig(model="fake")
    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True))
    store = Store.open(project.metadata_db_path)
    try:
        store.set_meta("index_state", "failed")
        store.set_meta("last_index_error", "oops")
    finally:
        store.close()

    with pytest.raises(SearchError, match="failed"):
        SearchEngine(fake_runtime).search_project(project, app_config, SearchOptions(query="before"), FreshnessMode.skip)


def test_search_project_auto_rebuilds_when_missing(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    project = _project(tmp_path)
    (tmp_path / "a.py").write_text("fresh auto", encoding="utf-8")
    app_config = AppConfig(model="fake")
    outcome = SearchEngine(fake_runtime).search_project(project, app_config, SearchOptions(query="fresh auto"), FreshnessMode.auto)
    assert outcome.results


class _CountingEmbedder:
    def __init__(self) -> None:
        self.ndim = 4
        self.query_calls = 0

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int | None = None) -> np.ndarray:
        import numpy as np

        del batch_size
        if is_query:
            self.query_calls += 1
        vectors = np.zeros((len(texts), self.ndim), dtype=np.float32)
        for i, text in enumerate(texts):
            n = float(len(text) or 1)
            vectors[i] = np.array([n, n % 7, n % 13, 1.0], dtype=np.float32)
        return vectors


class _CountingVectorIndex(FakeVectorIndex):
    def __init__(self, ndim: int = 4) -> None:
        super().__init__(ndim)
        self.load_calls = 0

    def load(self, path: Path) -> None:
        self.load_calls += 1
        super().load(path)


class _CountingRuntime:
    def __init__(self) -> None:
        from tests.conftest import FakeTextExtractor, FakeTokenChunker

        self.extractor = FakeTextExtractor()
        self.chunker = FakeTokenChunker()
        self.embedder = _CountingEmbedder()
        self.created_indexes: list[_CountingVectorIndex] = []

    def new_vector_index(self, ndim: int) -> _CountingVectorIndex:
        index = _CountingVectorIndex(ndim)
        self.created_indexes.append(index)
        return index


def test_empty_persistent_index_returns_no_results_without_query_embedding(tmp_path: Path) -> None:
    project = _project(tmp_path)
    app_config = AppConfig(model="fake")
    runtime = _CountingRuntime()

    # Create persistent artifacts with empty DB + empty vector index.
    IndexEngine(runtime).index_project(project, app_config, IndexOptions(rebuild=True))

    outcome = SearchEngine(runtime).search_project(project, app_config, SearchOptions(query="anything"), FreshnessMode.skip)
    assert outcome.results == []
    assert runtime.embedder.query_calls == 0


def test_missing_vector_index_with_existing_db_raises_actionable_error(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    project = _project(tmp_path)
    app_config = AppConfig(model="fake")
    (tmp_path / "a.py").write_text("hello vector", encoding="utf-8")
    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True))
    project.vector_index_path.unlink()

    with pytest.raises(SearchError, match="Vector index not found") as err:
        SearchEngine(fake_runtime).search_project(project, app_config, SearchOptions(query="hello"), FreshnessMode.skip)
    assert err.value.hint is not None


def test_missing_db_with_existing_vector_index_raises_actionable_error(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    project = _project(tmp_path)
    app_config = AppConfig(model="fake")
    (tmp_path / "a.py").write_text("hello db", encoding="utf-8")
    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True))
    project.metadata_db_path.unlink()

    with pytest.raises(SearchError, match="Persistent database not found") as err:
        SearchEngine(fake_runtime).search_project(project, app_config, SearchOptions(query="hello"), FreshnessMode.skip)
    assert err.value.hint is not None


def test_freshness_auto_missing_artifact_rebuilds(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    project = _project(tmp_path)
    app_config = AppConfig(model="fake")
    (tmp_path / "a.py").write_text("rebuild me", encoding="utf-8")
    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True))
    project.vector_index_path.unlink()

    outcome = SearchEngine(fake_runtime).search_project(project, app_config, SearchOptions(query="rebuild"), FreshnessMode.auto)
    assert outcome.results
    assert project.vector_index_path.exists()


def test_freshness_auto_stale_artifacts_do_incremental_index(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    project = _project(tmp_path)
    app_config = AppConfig(model="fake")
    (tmp_path / "a.py").write_text("old content", encoding="utf-8")
    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True))

    (tmp_path / "a.py").write_text("new content", encoding="utf-8")
    SearchEngine(fake_runtime).search_project(project, app_config, SearchOptions(query="new"), FreshnessMode.auto)

    outcome = SearchEngine(fake_runtime).search_project(project, app_config, SearchOptions(query="new"), FreshnessMode.skip)
    assert outcome.results
    assert "new content" in outcome.results[0].chunk_text


def test_freshness_check_clean_index_does_not_mutate_artifacts(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    project = _project(tmp_path)
    app_config = AppConfig(model="fake")
    (tmp_path / "a.py").write_text("stable", encoding="utf-8")
    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True))

    db_mtime_before = project.metadata_db_path.stat().st_mtime_ns
    vec_mtime_before = project.vector_index_path.stat().st_mtime_ns

    SearchEngine(fake_runtime).search_project(project, app_config, SearchOptions(query="stable"), FreshnessMode.check)

    assert project.metadata_db_path.stat().st_mtime_ns == db_mtime_before
    assert project.vector_index_path.stat().st_mtime_ns == vec_mtime_before


def test_open_project_reuses_single_loaded_reader_for_multiple_queries(tmp_path: Path) -> None:
    runtime = _CountingRuntime()
    project = _project(tmp_path)
    app_config = AppConfig(model="fake")
    (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
    IndexEngine(runtime).index_project(project, app_config, IndexOptions(rebuild=True))

    engine = SearchEngine(runtime)
    with CorpusAccess(runtime).open_project(project, app_config, freshness=FreshnessMode.skip) as reader:
        r1, _ = engine.search_reader(reader, SearchOptions(query="alpha"))
        r2, _ = engine.search_reader(reader, SearchOptions(query="beta"))

    assert r1
    assert r2
    assert len(runtime.created_indexes) == 2  # one for indexing, one for the reader session
    assert runtime.created_indexes[-1].load_calls == 1


def test_reader_flow_does_not_reload_index_every_query(tmp_path: Path) -> None:
    runtime = _CountingRuntime()
    project = _project(tmp_path)
    app_config = AppConfig(model="fake")
    (tmp_path / "a.py").write_text("query one and query two", encoding="utf-8")
    IndexEngine(runtime).index_project(project, app_config, IndexOptions(rebuild=True))

    engine = SearchEngine(runtime)
    with CorpusAccess(runtime).open_project(project, app_config, freshness=FreshnessMode.skip) as reader:
        for q in ("query one", "query two", "query"):
            engine.search_reader(reader, SearchOptions(query=q))

    assert runtime.created_indexes[-1].load_calls == 1


def test_effective_candidate_top_increases_for_scope_and_glob_filters() -> None:
    base = effective_candidate_top(SearchOptions(query="q", top=5))
    scoped = effective_candidate_top(SearchOptions(query="q", top=5, scope_path=Path("src")))
    included = effective_candidate_top(SearchOptions(query="q", top=5, include_globs=("*.py",)))
    excluded = effective_candidate_top(SearchOptions(query="q", top=5, exclude_globs=("*.md",)))
    assert base == 200
    assert scoped >= 1000
    assert included >= 1000
    assert excluded >= 1000


def test_scope_filtered_semantic_can_fallback_to_scoped_lexical(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    in_scope = tmp_path / "in_scope.py"
    out_scope = tmp_path / "out_scope.py"
    in_scope.write_text("needle in scope", encoding="utf-8")
    out_scope.write_text("needle out scope", encoding="utf-8")
    project = _project(tmp_path)
    app_config = AppConfig(model="fake")
    IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True))

    outcome = SearchEngine(fake_runtime).search_project(
        project,
        app_config,
        SearchOptions(
            query="needle",
            top=5,
            candidate_top=1,
            lexical_top=10,
            scope_path=in_scope,
        ),
        FreshnessMode.skip,
    )
    assert outcome.results
    assert all(r.file_path == in_scope for r in outcome.results)


def test_scope_path_file_returns_only_that_file(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    target = tmp_path / "target.py"
    other = tmp_path / "other.py"
    target.write_text("alpha", encoding="utf-8")
    other.write_text("alpha", encoding="utf-8")
    project = _project(tmp_path)
    IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

    outcome = SearchEngine(fake_runtime).search_project(
        project,
        AppConfig(model="fake"),
        SearchOptions(query="alpha", scope_path=target, lexical_top=20),
        FreshnessMode.skip,
    )
    assert outcome.results
    assert all(r.file_path == target for r in outcome.results)


def test_scope_path_directory_returns_only_descendants(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    scoped_dir = tmp_path / "scoped"
    scoped_dir.mkdir()
    (scoped_dir / "a.py").write_text("alpha", encoding="utf-8")
    (tmp_path / "outside.py").write_text("alpha", encoding="utf-8")
    project = _project(tmp_path)
    IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

    outcome = SearchEngine(fake_runtime).search_project(
        project,
        AppConfig(model="fake"),
        SearchOptions(query="alpha", scope_path=scoped_dir, lexical_top=20),
        FreshnessMode.skip,
    )
    assert outcome.results
    assert all(str(r.file_path).startswith(str(scoped_dir)) for r in outcome.results)


def test_outside_scope_results_excluded_from_semantic_and_lexical(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    scoped_dir = tmp_path / "scoped"
    scoped_dir.mkdir()
    in_file = scoped_dir / "inside.py"
    out_file = tmp_path / "outside.py"
    in_file.write_text("needle", encoding="utf-8")
    out_file.write_text("needle", encoding="utf-8")
    project = _project(tmp_path)
    IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

    outcome_sem = SearchEngine(fake_runtime).search_project(
        project,
        AppConfig(model="fake"),
        SearchOptions(query="needle", scope_path=scoped_dir, lexical_top=0, lexical_weight=0.0),
        FreshnessMode.skip,
    )
    assert outcome_sem.results
    assert all(r.file_path == in_file for r in outcome_sem.results)

    outcome_lex = SearchEngine(fake_runtime).search_project(
        project,
        AppConfig(model="fake"),
        SearchOptions(query="needle", scope_path=scoped_dir, candidate_top=1, lexical_top=20),
        FreshnessMode.skip,
    )
    assert outcome_lex.results
    assert all(r.file_path == in_file for r in outcome_lex.results)


def test_skip_freshness_without_artifacts_raises_persistent_index_not_found(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    project = _project(tmp_path)
    app_config = AppConfig(model="fake")
    with pytest.raises(SearchError, match="Persistent index not found"):
        SearchEngine(fake_runtime).search_project(project, app_config, SearchOptions(query="q"), FreshnessMode.skip)
