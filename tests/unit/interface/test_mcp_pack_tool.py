"""Unit tests for the MCP `pack` tool: pinned CLI-json payload shape
(parity rule: MCP payload == `simgrep pack --format json`), registry pin at
nine tools, schema bounds, and error paths — all without touching a real
model or index build."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from simgrep.config import save_app_config
from simgrep.mcp_server import handle_line
from simgrep.models import AppConfig, FileRole, SearchOutcome, SearchResult
from simgrep.pack import estimate_tokens

# ----------------------------------------------------------------------- helpers


def _rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
    request = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    response = handle_line(json.dumps(request))
    assert response is not None
    parsed: dict[str, Any] = json.loads(json.dumps(response))
    return parsed


def _call_pack(arguments: Any) -> dict[str, Any]:
    return _rpc("tools/call", {"name": "pack", "arguments": arguments})


def _result_text(response: dict[str, Any]) -> str:
    result = response["result"]
    return "".join(block["text"] for block in result["content"] if block.get("type") == "text")


def _payload(response: dict[str, Any]) -> dict[str, Any]:
    assert response["result"]["isError"] is False
    payload: dict[str, Any] = json.loads(_result_text(response))
    return payload


_PAYLOAD_KEYS = {"queries", "budget_tokens", "used_tokens", "pool_size", "dropped", "selections"}
_SELECTION_KEYS = {"path", "line_start", "line_end", "score", "tokens", "truncated", "text"}

_REGISTRY = {
    "search",
    "similar",
    "clusters",
    "status",
    "index",
    "diff",
    "expand",
    "pack",
    "debt",
}

# ----------------------------------------------------------------------- fake search seam


class FakePackEngine:
    """Duck-typed SearchEngine returning canned hits per query; records calls."""

    calls: list[tuple[str, int]] = []
    results_by_query: dict[str, list[SearchResult]] = {}
    base_path: Path = Path(".")

    def search_project(self, project: Any, app_config: Any, options: Any, freshness: Any) -> SearchOutcome:
        type(self).calls.append((options.query, options.top))
        return SearchOutcome(
            results=list(type(self).results_by_query.get(options.query, [])),
            base_path=type(self).base_path,
        )


def _hit(label: int, score: float, file_path: Path, text: str, line_start: int, line_end: int) -> SearchResult:
    return SearchResult(
        label=label,
        score=score,
        file_path=file_path,
        chunk_text=text,
        start_char=0,
        end_char=len(text),
        line_start=line_start,
        line_end=line_end,
        file_role=FileRole.source,
        language="python",
    )


@pytest.fixture()
def fake_pack(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolate HOME config and swap the lazy search seams for the fake engine."""
    home_config = tmp_path / "home" / ".config" / "simgrep"
    home_config.mkdir(parents=True)
    save_app_config(AppConfig(model="fake"), home_config / "config.toml")
    monkeypatch.setenv("HOME", str(home_config.parents[1]))

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    src = corpus / "payment.py"
    src.write_text("def charge_order(items):\n    receipt.charge()\n", encoding="utf-8")
    billing = corpus / "billing.py"
    billing.write_text("def bill(order):\n    return charge(order)\n", encoding="utf-8")

    t1 = "receipt.charge()"  # 16 chars -> 4 tokens
    t2 = "total = sum(items)"  # 18 chars -> 5 tokens
    t3 = "return charge(order)"  # 20 chars -> 5 tokens
    FakePackEngine.results_by_query = {
        "charge": [_hit(1, 0.9, src, t1, 2, 2), _hit(2, 0.85, src, t2, 1, 1)],
        # Same chunk as label 1 but a lower score: union must keep max score 0.9.
        "billing": [_hit(3, 0.8, billing, t3, 2, 2), _hit(1, 0.7, src, t1, 2, 2)],
    }
    FakePackEngine.base_path = corpus
    FakePackEngine.calls = []

    import simgrep.execution as runtime_module
    import simgrep.project as project_module
    import simgrep.search as search_module

    class FakeActive:
        model = "other-model"
        chunk_size = 64
        chunk_overlap = 8
        indexed_paths = (corpus,)

    class FakeRuntimeFactory:
        def for_app(self, app_config: Any) -> Any:
            return object()

        def for_project(self, project: Any) -> Any:
            return object()

    def fake_engine_ctor(runtime: Any) -> FakePackEngine:
        engine = FakePackEngine()
        return engine

    monkeypatch.setattr(project_module, "find_active_project", lambda path: FakeActive())
    monkeypatch.setattr(search_module, "SearchEngine", fake_engine_ctor)
    monkeypatch.setattr(runtime_module, "RuntimeFactory", FakeRuntimeFactory)
    return src


# ----------------------------------------------------------------------- registry


def test_registry_includes_pack_and_debt_is_last() -> None:
    tools = _rpc("tools/list", {})["result"]["tools"]
    assert len(tools) == 9  # debt: ninth and final tool
    by_name = {tool["name"]: tool for tool in tools}
    assert set(by_name) == _REGISTRY
    assert [tool["name"] for tool in tools][-1:] == ["debt"]
    spec = by_name["pack"]
    assert "pack" in spec["description"].lower()
    assert "budget" in spec["description"].lower()


def test_pack_schema_pins_bounds_and_required_arguments() -> None:
    tools = _rpc("tools/list", {})["result"]["tools"]
    schema = next(tool for tool in tools if tool["name"] == "pack")["inputSchema"]
    assert schema["type"] == "object"
    assert schema["required"] == ["queries"]
    props = schema["properties"]
    assert props["queries"]["type"] == "array"
    assert props["queries"]["items"] == {"type": "string"}
    assert props["queries"]["minItems"] == 1
    assert props["budget"]["minimum"] == 100
    assert props["budget"]["maximum"] == 200000
    assert props["per_query"]["minimum"] == 1
    assert props["per_query"]["maximum"] == 50
    assert props["lam"]["minimum"] == 0
    assert props["lam"]["maximum"] == 1
    assert props["path"]["type"] == "string"


@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        ({}, "Missing required argument 'queries'"),
        ({"path": "."}, "Missing required argument 'queries'"),
        ({"queries": "charge"}, "'queries' must be of type array"),
        ({"queries": [1]}, "'queries[0]' must be of type string"),
        ({"queries": ["charge"], "budget": 99}, "'budget' must be >="),
        ({"queries": ["charge"], "budget": 200001}, "'budget' must be <="),
        ({"queries": ["charge"], "budget": "big"}, "'budget' must be of type integer"),
        ({"queries": ["charge"], "per_query": 0}, "'per_query' must be >="),
        ({"queries": ["charge"], "per_query": 51}, "'per_query' must be <="),
        ({"queries": ["charge"], "lam": -0.1}, "'lam' must be >="),
        ({"queries": ["charge"], "lam": 1.5}, "'lam' must be <="),
    ],
)
def test_schema_violations_are_tool_errors(arguments: dict[str, Any], fragment: str) -> None:
    """SEP-1303: input validation failures are isError results, never tracebacks."""
    response = _call_pack(arguments)
    assert response["result"]["isError"] is True
    assert fragment in _result_text(response)


# ----------------------------------------------------------------------- payload parity


def test_happy_path_payload_matches_pinned_cli_json_contract(fake_pack: Path) -> None:
    response = _call_pack({"queries": ["charge", "billing"], "path": str(fake_pack.parent)})
    payload = _payload(response)

    assert set(payload) == _PAYLOAD_KEYS
    assert payload["queries"] == ["charge", "billing"]
    assert payload["budget_tokens"] == 3000
    assert payload["pool_size"] == 3  # label 1 deduped across queries
    assert payload["dropped"] == 0

    assert len(payload["selections"]) == 3
    for selection in payload["selections"]:
        assert set(selection) == _SELECTION_KEYS
    # Greedy MMR pick order under default lam=0.7: score-descending here.
    # Parity rule: paths are base-relative (CLI relativizes against outcome.base_path).
    assert [s["path"] for s in payload["selections"]] == ["payment.py", "payment.py", "billing.py"]
    scores = [s["score"] for s in payload["selections"]]
    assert scores == [0.9, 0.85, 0.8]
    assert all(s["truncated"] is False for s in payload["selections"])
    assert payload["used_tokens"] == sum(s["tokens"] for s in payload["selections"])
    assert payload["used_tokens"] <= payload["budget_tokens"]
    assert payload["selections"][0]["text"] == "receipt.charge()"
    assert payload["selections"][0]["tokens"] == estimate_tokens("receipt.charge()")


def test_handler_runs_one_search_per_query_with_pool_size(fake_pack: Path) -> None:
    _call_pack({"queries": ["charge", "billing"], "path": str(fake_pack.parent), "per_query": 8})
    assert FakePackEngine.calls == [("charge", 8), ("billing", 8)]


def test_label_dedup_keeps_max_score_across_queries(fake_pack: Path) -> None:
    payload = _payload(_call_pack({"queries": ["charge", "billing"], "path": str(fake_pack.parent)}))
    label_one_scores = [s["score"] for s in payload["selections"] if s["text"] == "receipt.charge()"]
    assert label_one_scores == [0.9]


def test_missing_line_info_falls_back_to_one_like_the_cli(fake_pack: Path) -> None:
    """Parity: CLI maps line_start/line_end None to 1; MCP must match, not 0."""
    FakePackEngine.results_by_query = {
        "nolines": [_hit(31, 0.9, fake_pack.parent / "payment.py", "receipt.charge()", None, None)],  # type: ignore[arg-type]
    }
    payload = _payload(_call_pack({"queries": ["nolines"], "path": str(fake_pack.parent)}))
    selection = payload["selections"][0]
    assert selection["line_start"] == 1
    assert selection["line_end"] == 1


def test_handler_uses_the_pack_engine_with_caller_tuning(monkeypatch: pytest.MonkeyPatch, fake_pack: Path) -> None:
    import simgrep.pack as pack_module

    observed: list[dict[str, Any]] = []
    real_pack_candidates = pack_module.pack_candidates

    def spy(candidates: Any, budget: Any, **kwargs: Any) -> Any:
        observed.append({"budget": budget, "lam": kwargs.get("lam"), "count": len(list(candidates))})
        return real_pack_candidates(candidates, budget, **kwargs)

    monkeypatch.setattr(pack_module, "pack_candidates", spy)
    _call_pack(
        {
            "queries": ["charge"],
            "path": str(fake_pack.parent),
            "budget": 500,
            "per_query": 2,
            "lam": 0.4,
        }
    )
    assert observed == [{"budget": 500, "lam": 0.4, "count": 2}]


def test_budget_drop_path_never_exceeds_budget(fake_pack: Path) -> None:
    big_a = "a" * 200  # 50 tokens
    big_b = "b" * 240  # 60 tokens
    FakePackEngine.results_by_query = {
        "big": [_hit(11, 0.95, fake_pack, big_a, 1, 4), _hit(12, 0.90, fake_pack, big_b, 5, 10)],
    }
    payload = _payload(_call_pack({"queries": ["big"], "path": str(fake_pack.parent), "budget": 100}))
    assert payload["pool_size"] == 2
    assert len(payload["selections"]) == 1
    assert payload["dropped"] == 1
    assert payload["used_tokens"] == 50
    assert payload["used_tokens"] <= payload["budget_tokens"]
    assert payload["selections"][0]["text"] == big_a


def test_oversized_fallback_truncates_to_budget_and_marks_it(fake_pack: Path) -> None:
    huge = "x" * 1000  # 250 tokens, nothing else in the pool
    FakePackEngine.results_by_query = {"huge": [_hit(21, 0.99, fake_pack, huge, 1, 40)]}
    payload = _payload(_call_pack({"queries": ["huge"], "path": str(fake_pack.parent), "budget": 100}))
    assert payload["pool_size"] == 1
    assert payload["dropped"] == 0
    assert len(payload["selections"]) == 1
    selection = payload["selections"][0]
    assert selection["truncated"] is True
    assert selection["text"].endswith("…")
    assert selection["tokens"] == 100
    assert payload["used_tokens"] == payload["budget_tokens"] == 100


def test_empty_pool_is_a_clean_tool_error_not_a_traceback(fake_pack: Path) -> None:
    FakePackEngine.results_by_query = {"ghost": []}
    response = _call_pack({"queries": ["ghost"], "path": str(fake_pack.parent)})
    assert response["result"]["isError"] is True
    assert "no candidates" in _result_text(response).lower()


# ----------------------------------------------------------------------- argument errors


@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        ({"queries": []}, "at least one"),
        ({"queries": ["   "]}, "at least one"),
        ({"queries": ["charge", "  "]}, "at least one"),
    ],
)
def test_empty_or_blank_queries_are_tool_errors(arguments: dict[str, Any], fragment: str) -> None:
    """The JSON-Schema subset validator ignores minItems; the handler still rejects cleanly."""
    response = _call_pack(arguments)
    assert response["result"]["isError"] is True
    assert fragment in _result_text(response)
