"""Regression: a rerank request must establish the torch-first libomp order
BEFORE the in-process search fallback constructs any USearchIndex.

usearch and torch each bundle an OpenMP runtime; whichever loads first owns
the process. ``search --rerank`` used to construct/load the vector index
first and only then import CrossEncoderReranker (-> sentence_transformers ->
torch), landing torch's libomp after usearch's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from simgrep.execution import RerankRequest, execute_search
from simgrep.models import AppConfig, SearchOptions
from tests.conftest import FakeRuntime, FakeVectorIndex


class _RecordingRuntime(FakeRuntime):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self._events = events

    def new_vector_index(self, ndim: int) -> FakeVectorIndex:
        self._events.append("index_constructed")
        return FakeVectorIndex(ndim)


def test_rerank_marks_torch_pending_before_index_construction(tmp_path: Path, monkeypatch: Any) -> None:
    events: list[str] = []

    class _PatchFactory:
        """Stub RuntimeFactory returning the recording runtime (canonical seam)."""

        def for_app(self, app_config: AppConfig) -> Any:
            return _RecordingRuntime(events)

    monkeypatch.setattr("simgrep.adapters.vector.mark_torch_pending", lambda: events.append("torch_pending"))
    monkeypatch.setattr("simgrep.execution.RuntimeFactory", _PatchFactory)
    (tmp_path / "a.py").write_text("rollback payment", encoding="utf-8")

    outcome = execute_search(
        app_config=AppConfig(model="fake"),
        path=tmp_path,
        options=SearchOptions(query="rollback"),
        ephemeral=True,
        rerank=RerankRequest(top=2, model="fake"),
    )

    assert outcome.results
    assert "torch_pending" in events, "rerank request never marked torch pending"
    assert events.index("torch_pending") < events.index("index_constructed"), f"torch pending flag arrived after index construction: {events}"
