"""Integration coverage for the fifth MCP tool (`clusters`)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from simgrep.config import save_app_config
from simgrep.indexing import IndexEngine
from simgrep.mcp_server import handle_message
from simgrep.models import AppConfig, IndexOptions, ProjectConfig
from simgrep.project import find_active_project, init_project
from tests.conftest import FakeTextExtractor, FakeTokenChunker, FakeVectorIndex

DUP_BODY = (
    "def transfer_funds(account_src, account_dst, amount):\n"
    "    ledger = open_ledger_connection()\n"
    "    ledger.debit(account_src, amount)\n"
    "    ledger.credit(account_dst, amount)\n"
)
OTHER_BODY = "class WeatherStationReader:\n" "    def calibrate_barometer(self):\n" "        return 'ok'\n"


class HashingEmbedder:
    """Deterministic token-set embedder: identical texts collide at cosine 1.0,
    disjoint token sets stay near-orthogonal (length-based FakeEmbedder cannot)."""

    ndim = 64

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int | None = None) -> np.ndarray:
        del is_query, batch_size
        vectors = np.zeros((len(texts), self.ndim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:4], "little"))
                vectors[row] += rng.standard_normal(self.ndim).astype(np.float32)
            norm = float(np.linalg.norm(vectors[row]))
            if norm > 0:
                vectors[row] /= norm
        return vectors


class RoundTripVectorIndex(FakeVectorIndex):
    """Preserves vectors through save()/load() instead of collapsing to ones."""

    def save(self, path: Path) -> None:
        payload = {str(label): vector.tolist() for label, vector in sorted(self.data.items())}
        path.write_text(json.dumps(payload), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        self.data = {int(label): np.asarray(vector, dtype=np.float32) for label, vector in json.loads(path.read_text(encoding="utf-8")).items()}


class HashingRuntime:
    def __init__(self) -> None:
        self.extractor = FakeTextExtractor()
        self.chunker = FakeTokenChunker()
        self.embedder = HashingEmbedder()

    def new_vector_index(self, ndim: int) -> RoundTripVectorIndex:
        return RoundTripVectorIndex(ndim)


class HashingRuntimeFactory:
    def __init__(self, runtime: HashingRuntime) -> None:
        self._runtime = runtime

    def for_app(self, config: AppConfig) -> HashingRuntime:
        del config
        return self._runtime

    def for_project(self, config: ProjectConfig) -> HashingRuntime:
        del config
        return self._runtime


def _isolate_global_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str = "fake") -> None:
    config_file = tmp_path / "home" / ".config" / "simgrep" / "config.toml"
    config_file.parent.mkdir(parents=True)
    save_app_config(AppConfig(model=model), config_file)
    monkeypatch.setenv("HOME", str(config_file.parents[2]))


@pytest.fixture
def clusters_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _isolate_global_config(tmp_path, monkeypatch)
    runtime = HashingRuntime()
    monkeypatch.setattr("simgrep.execution.RuntimeFactory", lambda: HashingRuntimeFactory(runtime))
    (tmp_path / "dup_a.py").write_text(DUP_BODY, encoding="utf-8")
    (tmp_path / "dup_b.py").write_text(DUP_BODY, encoding="utf-8")
    (tmp_path / "other.py").write_text(OTHER_BODY, encoding="utf-8")
    app_config = AppConfig(model="fake")
    init_project(tmp_path, app_config)
    project = find_active_project(tmp_path)
    assert project is not None
    IndexEngine(runtime).index_project(project, app_config, IndexOptions(rebuild=True))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _call_tool(name: str, arguments: dict[str, Any] | None = None, request_id: int = 1) -> dict[str, Any]:
    params: dict[str, Any] = {"name": name}
    if arguments is not None:
        params["arguments"] = arguments
    response = handle_message({"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": params})
    assert response is not None
    result = response["result"]
    assert isinstance(result, dict)
    assert len(result["content"]) == 1
    assert result["content"][0]["type"] == "text"
    return {"isError": bool(result["isError"]), "text": str(result["content"][0]["text"])}


def _payload(call: dict[str, Any]) -> list[dict[str, Any]]:
    assert not call["isError"]
    parsed = json.loads(call["text"])
    assert isinstance(parsed, list)
    return parsed


def test_clusters_tool_returns_relative_posix_member_payload(clusters_project: Path) -> None:
    records = _payload(_call_tool("clusters"))

    assert len(records) >= 1
    for record in records:
        assert set(record) == {"score", "duplicated_lines", "members"}
        assert 0 < record["score"] <= 1 + 1e-6  # float32 rounding can nudge an exact match past 1.0
        assert record["duplicated_lines"] >= 1
        for member in record["members"]:
            assert set(member) == {"label", "file_path", "line_start", "line_end"}
            assert isinstance(member["label"], int)
            shown = member["file_path"]
            assert not shown.startswith("/") and "\\" not in shown
            assert (clusters_project / shown).exists()
            assert 1 <= member["line_start"] <= member["line_end"]

    duplicate = next(r for r in records if len(r["members"]) >= 2)
    names = {Path(m["file_path"]).name for m in duplicate["members"]}
    assert names == {"dup_a.py", "dup_b.py"}
    assert all("other.py" not in {Path(m["file_path"]).name for m in r["members"]} for r in records)


def test_clusters_tool_reports_bad_arguments_as_tool_errors(clusters_project: Path) -> None:
    assert _call_tool("clusters", {"min_size": 1})["isError"]
    assert _call_tool("clusters", {"threshold": 1.5})["isError"]
    assert _call_tool("clusters", {"top": 0})["isError"]

    unknown = handle_message({"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "nope"}})
    assert unknown is not None and "error" in unknown
