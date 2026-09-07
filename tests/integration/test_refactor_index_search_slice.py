from __future__ import annotations

from pathlib import Path

from simgrep.indexing import IndexEngine
from simgrep.models import SCHEMA_VERSION, AppConfig, FreshnessMode, IndexOptions, ProjectConfig, ResultFilters, SearchOptions
from simgrep.search import SearchEngine
from simgrep.store import Store
from tests.conftest import FakeRuntime


def _project(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path,), "fake", 128, 20)


class TestRebuildConsistency:
    def test_rebuild_all_chunks_have_vector_keys(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        (tmp_path / "a.py").write_text("print(1)", encoding="utf-8")
        (tmp_path / "b.py").write_text("def foo(): pass", encoding="utf-8")
        project = _project(tmp_path)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

        store = Store.open(project.metadata_db_path)
        try:
            chunk_labels = {row[0] for row in store._conn.execute("SELECT label FROM chunks").fetchall()}
        finally:
            store.close()

        index = fake_runtime.new_vector_index(4)
        index.load(project.vector_index_path)
        vector_keys = set(index.data.keys())

        assert chunk_labels == vector_keys, f"Chunks have {chunk_labels} but vectors have {vector_keys}"

    def test_rebuild_max_label_consistent_with_chunks(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        (tmp_path / "a.py").write_text("alpha beta gamma", encoding="utf-8")
        (tmp_path / "b.py").write_text("delta epsilon", encoding="utf-8")
        project = _project(tmp_path)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

        store = Store.open(project.metadata_db_path)
        try:
            chunk_labels = [row[0] for row in store._conn.execute("SELECT label FROM chunks ORDER BY label").fetchall()]
            max_label_str = store.get_meta("max_label")
            assert max_label_str is not None
            max_label = int(max_label_str)
        finally:
            store.close()

        assert chunk_labels, "No chunks found"
        assert max_label == chunk_labels[-1], f"max_label={max_label} but last chunk label={chunk_labels[-1]}"


class TestIncrementalDelete:
    def test_incremental_delete_no_orphan_vectors(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        (tmp_path / "a.py").write_text("print(1)", encoding="utf-8")
        (tmp_path / "b.py").write_text("def foo(): pass", encoding="utf-8")
        project = _project(tmp_path)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

        index_before = fake_runtime.new_vector_index(4)
        index_before.load(project.vector_index_path)
        vector_count_before = len(index_before.data)

        (tmp_path / "a.py").unlink()
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions())

        index_after = fake_runtime.new_vector_index(4)
        index_after.load(project.vector_index_path)
        vector_count_after = len(index_after.data)

        store = Store.open(project.metadata_db_path)
        try:
            chunk_count = store.counts().chunks_count
        finally:
            store.close()

        assert chunk_count == vector_count_after, f"Chunks={chunk_count} but vectors={vector_count_after}"
        assert vector_count_after < vector_count_before, "Expected fewer vectors after delete"


class TestIncrementalChange:
    def test_incremental_change_no_stale_chunks_terms(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
        project = _project(tmp_path)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

        (tmp_path / "a.py").write_text("gamma delta", encoding="utf-8")
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions())

        store = Store.open(project.metadata_db_path)
        try:
            chunks = store._conn.execute("SELECT label, text FROM chunks").fetchall()
            terms = store._conn.execute("SELECT label, term FROM terms").fetchall()
        finally:
            store.close()

        chunk_texts = {row[1] for row in chunks}
        assert "alpha beta" not in chunk_texts, "Stale chunk text found after change"
        assert "gamma delta" in chunk_texts, "New chunk text not found after change"

        term_texts = {row[1] for row in terms}
        assert "alpha" not in term_texts and "beta" not in term_texts, "Stale terms found after change"
        assert "gamma" in term_texts and "delta" in term_texts, "New terms not found after change"


class TestLexicalSearchAfterDelete:
    def test_lexical_search_no_stale_terms_after_delete(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
        project = _project(tmp_path)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

        store = Store.open(project.metadata_db_path)
        try:
            before = store.lexical_candidates(["alpha"], limit=10, filters=ResultFilters())
            assert len(before) == 1
        finally:
            store.close()

        (tmp_path / "a.py").unlink()
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions())

        store = Store.open(project.metadata_db_path)
        try:
            after = store.lexical_candidates(["alpha"], limit=10, filters=ResultFilters())
            assert len(after) == 0, f"Found stale terms after delete: {after}"
        finally:
            store.close()


class TestTermStatsUpdate:
    def test_term_stats_updated_after_add(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        project = _project(tmp_path)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions())

        (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions())

        store = Store.open(project.metadata_db_path)
        try:
            store.refresh_term_stats()
            stats = store._conn.execute("SELECT term, chunk_df FROM term_stats WHERE term = 'alpha'").fetchone()
            assert stats is not None and stats[1] == 1, f"term_stats not updated after add: {stats}"
        finally:
            store.close()

    def test_term_stats_updated_after_change(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
        project = _project(tmp_path)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

        (tmp_path / "a.py").write_text("alpha gamma", encoding="utf-8")
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions())

        store = Store.open(project.metadata_db_path)
        try:
            store.refresh_term_stats()
            beta_stats = store._conn.execute("SELECT chunk_df FROM term_stats WHERE term = 'beta'").fetchone()
            gamma_stats = store._conn.execute("SELECT chunk_df FROM term_stats WHERE term = 'gamma'").fetchone()
            assert beta_stats is None or beta_stats[0] == 0, "Stale term 'beta' still in term_stats"
            assert gamma_stats is not None and gamma_stats[0] == 1, "New term 'gamma' not in term_stats"
        finally:
            store.close()

    def test_term_stats_updated_after_delete(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
        project = _project(tmp_path)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

        (tmp_path / "a.py").unlink()
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions())

        store = Store.open(project.metadata_db_path)
        try:
            store.refresh_term_stats()
            alpha_stats = store._conn.execute("SELECT chunk_df FROM term_stats WHERE term = 'alpha'").fetchone()
            assert alpha_stats is None or alpha_stats[0] == 0, "Deleted term 'alpha' still in term_stats"
        finally:
            store.close()


class TestScopePathFilters:
    def test_scope_path_filter_on_semantic(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
        (tmp_path / "b.py").write_text("alpha beta", encoding="utf-8")
        project = _project(tmp_path)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

        outcome = SearchEngine(fake_runtime).search_project(
            project,
            AppConfig(model="fake"),
            SearchOptions(query="alpha", top=10, min_score=0.0, scope_path=tmp_path / "a.py"),
            FreshnessMode.skip,
        )
        assert all(str(r.file_path).endswith("a.py") for r in outcome.results), f"Wrong scope filter on semantic: {[r.file_path for r in outcome.results]}"

    def test_scope_path_filter_on_lexical(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        (tmp_path / "a.py").write_text("alpha beta", encoding="utf-8")
        (tmp_path / "b.py").write_text("alpha beta", encoding="utf-8")
        project = _project(tmp_path)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

        outcome = SearchEngine(fake_runtime).search_project(
            project,
            AppConfig(model="fake"),
            SearchOptions(query="alpha", top=10, min_score=0.0, lexical_top=10, scope_path=tmp_path / "a.py"),
            FreshnessMode.skip,
        )
        assert all(str(r.file_path).endswith("a.py") for r in outcome.results), f"Wrong scope filter on lexical: {[r.file_path for r in outcome.results]}"


class TestLookupChunksOrder:
    def test_lookup_chunks_returns_in_vector_hits_order(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        (tmp_path / "a.py").write_text("first", encoding="utf-8")
        (tmp_path / "b.py").write_text("second", encoding="utf-8")
        project = _project(tmp_path)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))

        outcome = SearchEngine(fake_runtime).search_project(
            project,
            AppConfig(model="fake"),
            SearchOptions(query="first second", top=10, min_score=0.0, lexical_top=0),
            FreshnessMode.skip,
        )

        store = Store.open(project.metadata_db_path)
        try:
            labels = [r.label for r in outcome.results]
            chunks = store.lookup_chunks(labels, ResultFilters())
            chunk_labels = [c["label"] for c in chunks]
            assert chunk_labels == labels, f"lookup_chunks returned {chunk_labels} but expected {labels}"
        finally:
            store.close()


class TestHybridSymbolHeavy:
    def test_hybrid_beats_no_hybrid_on_symbol_heavy_query(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        (tmp_path / "a_generic.py").write_text("payment flow handler", encoding="utf-8")
        (tmp_path / "z_symbolic.py").write_text(
            "def fetch_user_by_id(user_id: str) -> dict:\n    return {}\nfetch_user_by_id(user_id='42')",
            encoding="utf-8",
        )
        project = _project(tmp_path)
        app_config = AppConfig(model="fake")
        IndexEngine(fake_runtime).index_project(project, app_config, IndexOptions(rebuild=True))

        no_hybrid = SearchEngine(fake_runtime).search_project(
            project,
            app_config,
            SearchOptions(
                query="fetch_user_by_id",
                top=5,
                lexical_top=0,
                lexical_weight=0.0,
            ),
            FreshnessMode.skip,
        )
        hybrid = SearchEngine(fake_runtime).search_project(
            project,
            app_config,
            SearchOptions(
                query="fetch_user_by_id",
                top=5,
                lexical_top=10,
                lexical_weight=0.5,
            ),
            FreshnessMode.skip,
        )

        assert no_hybrid.results
        assert hybrid.results
        assert no_hybrid.results[0].file_path.name == "a_generic.py"
        assert hybrid.results[0].file_path.name == "z_symbolic.py"
