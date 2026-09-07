from __future__ import annotations

from pathlib import Path

import pytest

from simgrep.adapters.extractor import TextExtractor
from simgrep.adapters.vector import USearchIndex
from simgrep.models import SCHEMA_VERSION, AppConfig, ProjectConfig
from simgrep.runtime import Runtime, RuntimeFactory


def test_runtime_factory_app_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = RuntimeFactory()
    runtime = Runtime(extractor=None, chunker=None, embedder=None)  # type: ignore[arg-type]
    monkeypatch.setattr(factory, "_runtime", lambda *_: runtime)
    first = factory.for_app(AppConfig(model="m"))
    second = factory.for_app(AppConfig(model="m"))
    assert first is second


def test_runtime_factory_project_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    factory = RuntimeFactory()
    runtime = Runtime(extractor=None, chunker=None, embedder=None)  # type: ignore[arg-type]
    monkeypatch.setattr(factory, "_runtime", lambda *_: runtime)
    project = ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path,), "m", 64, 8)
    first = factory.for_project(project)
    second = factory.for_project(project)
    assert first is second


def test_new_vector_index_uses_inner_product_metric() -> None:
    runtime = Runtime(extractor=None, chunker=None, embedder=None)  # type: ignore[arg-type]

    index = runtime.new_vector_index(ndim=5)

    assert isinstance(index, USearchIndex)
    assert index.metric == "ip"
    assert index.ndim == 5


def test_runtime_factory_builds_and_caches_per_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed: list[tuple[str, str, int]] = []

    class FakeChunker:
        def __init__(self, model_name: str, chunk_size: int, overlap: int) -> None:
            constructed.append(("chunker", model_name, chunk_size))

    class FakeEmbedder:
        def __init__(self, model_name: str, normalize_embeddings: bool) -> None:
            constructed.append(("embedder", model_name, int(normalize_embeddings)))

    monkeypatch.setattr("simgrep.runtime.HFChunker", FakeChunker)
    monkeypatch.setattr("simgrep.runtime.SentenceEmbedder", FakeEmbedder)

    factory = RuntimeFactory()
    first = factory.for_app(AppConfig(model="m1", chunk_size=64, chunk_overlap=8))
    second = factory.for_app(AppConfig(model="m1", chunk_size=64, chunk_overlap=8))
    other = factory.for_app(AppConfig(model="m2", chunk_size=64, chunk_overlap=8))

    assert first is second
    assert first is not other
    assert isinstance(first.extractor, TextExtractor)
    # The chunker is lazy: construction is deferred until first chunk/require,
    # so the factory itself must not have built either FakeChunker yet.
    assert constructed.count(("chunker", "m1", 64)) == 0
    first.chunker.require()
    assert constructed.count(("chunker", "m1", 64)) == 1
    first.chunker.require()
    assert constructed.count(("chunker", "m1", 64)) == 1, "chunker must materialize once"
    other.chunker.require()
    assert ("chunker", "m2", 64) in constructed
    # The bulk embedder is lazy, and the query session is lazy too (built on
    # first encode), so the factory itself materializes no embedder at all.
    assert constructed.count(("embedder", "m1", 1)) == 0
