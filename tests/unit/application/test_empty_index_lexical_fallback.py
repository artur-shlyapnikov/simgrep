from __future__ import annotations

from pathlib import Path

from simgrep.corpus import CorpusReader
from simgrep.indexing import IndexEngine
from simgrep.models import SCHEMA_VERSION, Anchor, AppConfig, EphemeralIndexOptions, FreshnessMode, IndexOptions, ProjectConfig, SearchOptions, SimilarOptions
from simgrep.search import SearchEngine
from tests.conftest import FakeRuntime


def _ephemeral_reader(tmp_path: Path, fake_runtime: FakeRuntime) -> tuple[SearchEngine, CorpusReader]:
    (tmp_path / "sample.py").write_text("alpha beta gamma\n", encoding="utf-8")
    engine = SearchEngine(fake_runtime)
    from simgrep.indexing import IndexEngine

    reader = IndexEngine(fake_runtime).build_ephemeral([tmp_path], AppConfig(model="fake"), EphemeralIndexOptions())
    # Simulate a crash-diverged index: store has chunks, vector index is empty.
    reader._index = fake_runtime.new_vector_index(fake_runtime.embedder.ndim)
    assert reader.chunk_count == 0
    return engine, reader


class TestEmptyIndexLexicalFallback:
    def test_search_reader_returns_lexical_results_when_index_empty(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        engine, reader = _ephemeral_reader(tmp_path, fake_runtime)
        try:
            results, semantic_count = engine.search_reader(reader, SearchOptions(query="alpha"))
            assert results, "expected lexical-only results from empty index"
            assert semantic_count == 0
            assert any("sample.py" == r.file_path.name for r in results)
        finally:
            reader.close()

    def test_similar_reader_returns_lexical_results_when_index_empty(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        engine, reader = _ephemeral_reader(tmp_path, fake_runtime)
        try:
            options = SimilarOptions(search=SearchOptions(query=""), anchor=Anchor(text="beta"))
            results, semantic_count = engine._similar_reader(reader, options)
            assert results, "expected lexical-only similar results from empty index"
            assert semantic_count == 0
            assert any("sample.py" == r.file_path.name for r in results)
        finally:
            reader.close()

    def test_similar_reader_unlike_anchor_degrades_to_lexical_when_index_empty(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        engine, reader = _ephemeral_reader(tmp_path, fake_runtime)
        try:
            options = SimilarOptions(
                search=SearchOptions(query=""),
                anchor=Anchor(text="beta"),
                unlike=Anchor(text="alpha"),
            )
            results, semantic_count = engine._similar_reader(reader, options)
            assert results, "expected lexical-only results even with an unlike anchor"
            assert semantic_count == 0
        finally:
            reader.close()


class TestPersistentProjectLexicalFallback:
    def test_search_project_returns_lexical_results_when_persistent_vector_index_empty(self, tmp_path: Path, fake_runtime: FakeRuntime) -> None:
        (tmp_path / "sample.py").write_text("alpha beta gamma\n", encoding="utf-8")
        project = ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path,), "fake", 128, 20)
        IndexEngine(fake_runtime).index_project(project, AppConfig(model="fake"), IndexOptions(rebuild=True))
        # Crash-diverge: store keeps its chunk, on-disk vectors are wiped.
        fake_runtime.new_vector_index(4).save(project.vector_index_path)

        outcome = SearchEngine(fake_runtime).search_project(project, AppConfig(model="fake"), SearchOptions(query="alpha"), FreshnessMode.skip)

        assert outcome.results, "expected lexical-only results through the persistent project path"
        assert outcome.semantic_candidates == 0
        assert any(r.file_path.name == "sample.py" for r in outcome.results)
