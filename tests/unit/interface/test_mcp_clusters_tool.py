"""Unit tests for the MCP `clusters` tool: pinned JSON payload shape, argument
validation conventions, and persistent/ephemeral project resolution — all over
faked engines so no embedding model is ever loaded."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from simgrep.clusters_engine import ClustersEngine as RealClustersEngine
from simgrep.config import save_app_config
from simgrep.mcp_server import handle_message
from simgrep.models import (
    AppConfig,
    ClusterMember,
    ClustersOptions,
    ClustersOutcome,
    ProjectConfig,
    SemanticCluster,
)

PROJECT_ROOT = Path("/tmp/fake-project")
CANNED_OUTCOME = ClustersOutcome(
    clusters=(
        SemanticCluster(
            members=(
                ClusterMember(label=17, file_path=str(PROJECT_ROOT / "src" / "a.py"), line_start=10, line_end=42),
                ClusterMember(label=23, file_path=str(PROJECT_ROOT / "src" / "b.py"), line_start=5, line_end=30),
            ),
            score=0.93,
            duplicated_lines=84,
        ),
    ),
    total_found=1,
    chunks_scanned=500,
)


def _project() -> ProjectConfig:
    return ProjectConfig(
        schema_version=1,
        name="fake",
        root=PROJECT_ROOT,
        indexed_paths=(PROJECT_ROOT,),
        model="fake",
        chunk_size=512,
        chunk_overlap=64,
    )


def _rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
    request = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    response = handle_message(request)
    assert response is not None
    return json.loads(response) if isinstance(response, str) else response


def _call_clusters(arguments: Any) -> dict[str, Any]:
    return _rpc("tools/call", {"name": "clusters", "arguments": arguments})


def _result_text(response: dict[str, Any]) -> str:
    result = response["result"]
    return "".join(block["text"] for block in result["content"] if block.get("type") == "text")


class FakeClustersEngine:
    """Duck-typed ClustersEngine returning a canned outcome; records calls."""

    mode: str = "clusters_project"

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.calls: list[tuple[str, ClustersOptions]] = []

    def run_batch(self, batch: Any, options: ClustersOptions) -> ClustersOutcome:
        del batch
        self.calls.append((FakeClustersEngine.mode, options))
        return CANNED_OUTCOME


class FakeRuntimeFactory:
    """Duck-typed RuntimeFactory handing back an inert sentinel runtime."""

    def for_app(self, config: AppConfig) -> object:
        del config
        return object()

    def for_project(self, config: ProjectConfig) -> object:
        del config
        return object()


@pytest.fixture()
def fake_engines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[FakeClustersEngine]:
    """Isolate HOME config and swap the lazy engine seams for fakes."""
    home_config = tmp_path / "home" / ".config" / "simgrep"
    home_config.mkdir(parents=True)
    save_app_config(AppConfig(model="fake"), home_config / "config.toml")
    monkeypatch.setenv("HOME", str(home_config.parents[1]))
    created: list[FakeClustersEngine] = []
    from contextlib import contextmanager as _cm

    import simgrep.clusters_engine as clusters_engine_module
    import simgrep.corpus as corpus_module

    def build(runtime: Any = None) -> FakeClustersEngine:
        instance = FakeClustersEngine(runtime)
        created.append(instance)
        return instance

    @_cm
    def _open_project(self: Any, project: Any, app_config: Any, *, freshness: Any = None) -> Iterator[_FakeReader]:
        FakeClustersEngine.mode = "clusters_project"
        yield _FakeReader()

    @_cm
    def _open_ephemeral(self: Any, paths: Any, app_config: Any, options: Any = None) -> Iterator[_FakeReader]:
        FakeClustersEngine.mode = "clusters_path"
        yield _FakeReader()

    class _FakeReader:
        def snapshot(self) -> str:
            return "batch"

        def counts(self, name: str = "") -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(corpus_module.CorpusAccess, "open_project", _open_project)
    monkeypatch.setattr(corpus_module.CorpusAccess, "open_ephemeral", _open_ephemeral)
    monkeypatch.setattr(clusters_engine_module, "ClustersEngine", build)
    monkeypatch.setattr("simgrep.execution.RuntimeFactory", FakeRuntimeFactory)
    return created


def test_tools_list_exposes_clusters_with_input_schema() -> None:
    tools = _rpc("tools/list", {})["result"]["tools"]
    clusters = next(tool for tool in tools if tool["name"] == "clusters")
    schema = clusters["inputSchema"]
    assert schema["type"] == "object"
    for prop in ("path", "threshold", "min_size", "top", "same_file"):
        assert prop in schema["properties"]
    assert schema["properties"]["path"]["type"] == "string"
    assert schema["properties"]["same_file"]["type"] == "boolean"


def test_call_returns_pinned_cluster_payload(fake_engines: list[FakeClustersEngine], monkeypatch: pytest.MonkeyPatch) -> None:
    active = _project()
    monkeypatch.setattr("simgrep.project.find_active_project", lambda cwd: active)
    response = _call_clusters({"threshold": 0.9, "min_size": 3, "top": 7, "same_file": True})
    assert response["result"]["isError"] is False
    payload = json.loads(_result_text(response))
    assert isinstance(payload, list) and len(payload) == 1
    cluster = payload[0]
    assert set(cluster) == {"score", "duplicated_lines", "members"}
    assert cluster["score"] == 0.93
    assert cluster["duplicated_lines"] == 84
    assert cluster["members"] == [
        {"label": 17, "file_path": "src/a.py", "line_start": 10, "line_end": 42},
        {"label": 23, "file_path": "src/b.py", "line_start": 5, "line_end": 30},
    ]
    engine = fake_engines[-1]
    assert [name for name, _ in engine.calls] == ["clusters_project"]
    options = engine.calls[0][1]
    assert isinstance(options, ClustersOptions)
    assert (options.threshold, options.min_size, options.top, options.same_file) == (0.9, 3, 7, True)


def test_uncovered_path_routes_through_clusters_path(fake_engines: list[FakeClustersEngine], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An uncovered path takes the engine's single canonical ephemeral path —
    no local IndexEngine composition, caller options forwarded untouched."""
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr("simgrep.project.find_active_project", lambda cwd: None)
    response = _call_clusters({"path": str(outside)})
    assert response["result"]["isError"] is False
    payload = json.loads(_result_text(response))
    assert payload[0]["members"][0]["file_path"].endswith("src/a.py")
    engine = fake_engines[-1]
    assert [name for name, _ in engine.calls] == ["clusters_path"]
    options = engine.calls[0][1]
    assert isinstance(options, ClustersOptions)
    assert (options.threshold, options.min_size, options.top, options.same_file) == (0.8, 2, 20, False)


def test_no_path_and_no_active_project_is_tool_error(fake_engines: list[FakeClustersEngine], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("simgrep.project.find_active_project", lambda cwd: None)
    response = _call_clusters({})
    assert response["result"]["isError"] is True
    assert "No active project found" in _result_text(response)


def test_wrong_type_threshold_is_tool_error(fake_engines: list[FakeClustersEngine]) -> None:
    response = _call_clusters({"threshold": "high"})
    assert response["result"]["isError"] is True
    assert "must be of type number" in _result_text(response)
    assert fake_engines == []


def test_min_size_below_two_is_tool_error(fake_engines: list[FakeClustersEngine]) -> None:
    response = _call_clusters({"min_size": 1})
    assert response["result"]["isError"] is True
    assert "must be >= 2" in _result_text(response)
    assert fake_engines == []


def test_threshold_zero_is_tool_error(fake_engines: list[FakeClustersEngine], monkeypatch: pytest.MonkeyPatch) -> None:
    """Threshold validity is a domain contract on ClustersEngine.run_batch."""
    from numpy import zeros

    from simgrep.corpus import ChunkBatch

    empty = ChunkBatch(chunks=(), vectors=zeros((0, 2), dtype="float32"), indexed_count=0)
    with pytest.raises(Exception) as exc:
        RealClustersEngine(None).run_batch(empty, ClustersOptions(threshold=0))
    assert "Threshold" in str(exc.value)


def test_non_object_arguments_is_protocol_error() -> None:
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "clusters", "arguments": ["not", "an", "object"]},
    }
    response = handle_message(request)
    assert response is not None
    response = json.loads(response) if isinstance(response, str) else response
    assert response["error"]["code"] == -32602


def test_unknown_tool_name_is_protocol_error() -> None:
    response = _rpc("tools/call", {"name": "definitely_not_a_tool", "arguments": {}})
    assert response["error"]["code"] == -32602
