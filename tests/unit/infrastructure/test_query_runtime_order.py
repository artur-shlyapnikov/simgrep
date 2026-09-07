"""Regression: persistent flows must resolve the lazy query embedder BEFORE
constructing or loading any USearchIndex.

With no local ONNX cache, ``_LazyQueryEmbedder`` falls back to the torch bulk
embedder mid-request; if that happens after a USearchIndex was built, torch's
libomp loads after usearch's (OpenMP segfault guard). Every USearchIndex in a
simgrep process is born in ``Runtime.new_vector_index``, so that seam owns the
invariant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from simgrep.corpus import _open_persistent_reader
from simgrep.models import SCHEMA_VERSION, ProjectConfig
from simgrep.runtime import Runtime
from simgrep.store import Store
from tests.conftest import FakeEmbedder, FakeTextExtractor, FakeTokenChunker


class _RecordingLazyQuery:
    """Duck-typed stand-in for _LazyQueryEmbedder with an observable require()."""

    ndim = 4

    def __init__(self, events: list[str]) -> None:
        self._events = events

    def require(self) -> "_RecordingLazyQuery":
        self._events.append("query_runtime_ensured")
        return self


def _recording_index(events: list[str]) -> type:
    class _RecordingUSearchIndex:
        def __init__(self, ndim: int, metric: str = "ip", dtype: str = "f32") -> None:
            del ndim, metric, dtype
            events.append("index_constructed")

        def load(self, path: Path) -> None:
            del path

    return _RecordingUSearchIndex


def _runtime(events: list[str]) -> Runtime:
    return Runtime(
        extractor=FakeTextExtractor(),  # type: ignore[arg-type]
        chunker=FakeTokenChunker(),
        embedder=FakeEmbedder(),
        query_embedder=_RecordingLazyQuery(events),  # type: ignore[arg-type]
    )


def test_new_vector_index_resolves_query_runtime_before_usearch(monkeypatch: Any) -> None:
    events: list[str] = []
    monkeypatch.setattr("simgrep.runtime.USearchIndex", _recording_index(events))

    _runtime(events).new_vector_index(4)

    assert events == ["query_runtime_ensured", "index_constructed"]


def test_persistent_reader_requires_query_runtime_before_index_open(tmp_path: Path, monkeypatch: Any) -> None:
    project = ProjectConfig(SCHEMA_VERSION, "p", tmp_path, (tmp_path,), "fake", 128, 20)
    project.simgrep_dir.mkdir(parents=True, exist_ok=True)
    Store.open(project.metadata_db_path).close()
    events: list[str] = []
    monkeypatch.setattr("simgrep.runtime.USearchIndex", _recording_index(events))

    with _open_persistent_reader(_runtime(events), project):
        pass

    assert events == ["query_runtime_ensured", "index_constructed"]


def test_runtime_ensure_query_runtime_resolves_lazy_query_embedder() -> None:
    events: list[str] = []

    class LazyQuery:
        def require(self) -> "LazyQuery":
            events.append("require")
            return self

    runtime = Runtime(extractor=FakeTextExtractor(), chunker=FakeTokenChunker(), embedder=FakeEmbedder(), query_embedder=LazyQuery())  # type: ignore[arg-type]
    runtime.ensure_query_runtime()
    assert events == ["require"]


def test_runtime_ensure_query_runtime_is_noop_for_eager_embedder() -> None:
    class EagerQuery:
        ndim = 4

    runtime = Runtime(extractor=FakeTextExtractor(), chunker=FakeTokenChunker(), embedder=FakeEmbedder(), query_embedder=EagerQuery())  # type: ignore[arg-type]
    # Eager embedders have no deferred resolution left; must not raise.
    runtime.ensure_query_runtime()
