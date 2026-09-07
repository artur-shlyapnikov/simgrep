"""Guard for the torch-before-usearch libomp invariant on server surfaces.

A served query leaves usearch (libomp) resident with torch unloaded; a later
in-process bulk index imports torch afterwards. ``assert_safe_bulk_entry``
must fire only when the order was already violated AND the OMP_NUM_THREADS=1
mitigation (simgrep/__init__.py) is absent; it must be a no-op in fresh
processes (usearch never loaded, CLI one-shot path).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from simgrep.adapters import vector as vector_mod
from simgrep.errors import SimgrepError
from simgrep.runtime import assert_safe_bulk_entry

_USEARCH_SENTINEL = object()


def _set_usearch_loaded(monkeypatch: pytest.MonkeyPatch, loaded: bool) -> None:
    monkeypatch.setattr(vector_mod, "_USEARCH_MODULE", _USEARCH_SENTINEL if loaded else None)


def test_fires_when_usearch_loaded_torch_absent_unmitigated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    _set_usearch_loaded(monkeypatch, loaded=True)
    with pytest.raises(SimgrepError, match="usearch"):
        assert_safe_bulk_entry()


def test_noop_when_mitigation_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    _set_usearch_loaded(monkeypatch, loaded=True)
    monkeypatch.delitem(sys.modules, "torch", raising=False)  # isolate worker ordering
    assert_safe_bulk_entry()  # documented mitigation holds
    assert "torch" not in sys.modules  # cheap check must not import torch


def test_noop_in_fresh_process_usearch_not_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    _set_usearch_loaded(monkeypatch, loaded=False)
    assert_safe_bulk_entry()


def test_noop_when_torch_already_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    _set_usearch_loaded(monkeypatch, loaded=True)

    class _FakeTorch:
        pass

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    assert_safe_bulk_entry()


@dataclass(frozen=True)
class _Stats:
    files_indexed: int = 0


def test_tool_index_invokes_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """The MCP index tool runs the guard before any indexing work."""
    from simgrep.tool_registry import _tool_index

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    _set_usearch_loaded(monkeypatch, loaded=True)
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("indexing must not start past a failed guard")

    monkeypatch.setattr("simgrep.indexing.IndexEngine", _boom)
    with pytest.raises(SimgrepError, match="usearch"):
        _tool_index({})


def test_tool_index_proceeds_under_mitigation(monkeypatch: pytest.MonkeyPatch) -> None:
    """With OMP_NUM_THREADS=1 the guard is transparent and indexing proceeds."""
    from simgrep.tool_registry import _tool_index

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    _set_usearch_loaded(monkeypatch, loaded=True)

    app_config = SimpleNamespace(model="m", chunk_size=1, chunk_overlap=2, max_chars=3)
    project = SimpleNamespace(model="other", chunk_size=9, chunk_overlap=9)
    runtime = object()
    seen: dict[str, object] = {}

    class _FakeEngine:
        def __init__(self, rt: object) -> None:
            seen["runtime"] = rt

        def index_project(self, proj: object, cfg: object, opts: object) -> _Stats:
            seen["project"] = proj
            return _Stats()

    class _FakeFactory:
        def for_project(self, proj: object) -> object:
            return runtime

    monkeypatch.setattr("simgrep.config.load_app_config", lambda: app_config)
    monkeypatch.setattr("simgrep.project.require_active_project", lambda: project)
    monkeypatch.setattr("simgrep.execution.factory", lambda: _FakeFactory())
    monkeypatch.setattr("simgrep.indexing.IndexEngine", _FakeEngine)

    payload = _tool_index({})
    assert payload == {"files_indexed": 0}
    assert seen["runtime"] is runtime
    assert seen["project"] is project
