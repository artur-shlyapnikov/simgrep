"""Unit tests for the MCP `debt` tool: pinned CLI-json payload shape
(parity rule: MCP payload == `simgrep debt --format json`), registry pin at
nine tools, schema bounds, persistent/ephemeral duality, and error paths —
all without touching a real model, index build, or git worktree."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from simgrep.config import save_app_config
from simgrep.errors import DebtError
from simgrep.mcp_server import handle_line
from simgrep.models import AppConfig, DebtMatch, DebtOptions, DebtReport, DebtTheme

# ----------------------------------------------------------------------- helpers


def _rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
    request = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    response = handle_line(json.dumps(request))
    assert response is not None
    parsed: dict[str, Any] = json.loads(json.dumps(response))
    return parsed


def _call_debt(arguments: dict[str, Any]) -> dict[str, Any]:
    return _rpc("tools/call", {"name": "debt", "arguments": arguments})


def _result_text(response: dict[str, Any]) -> str:
    result = response["result"]
    return "".join(block["text"] for block in result["content"] if block.get("type") == "text")


def _payload(response: dict[str, Any]) -> dict[str, Any]:
    assert response["result"]["isError"] is False
    payload: dict[str, Any] = json.loads(_result_text(response))
    return payload


_PAYLOAD_KEYS = {
    "themes",
    "scattered",
    "markers_found",
    "chunks_scanned",
    "truncated",
    "threshold",
    "max_age_days",
    "passed",
}
_THEME_KEYS = {"label", "size", "oldest_epoch", "matches"}
_MATCH_KEYS = {"file_path", "line_start", "marker", "snippet"}

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


# ----------------------------------------------------------------------- fake engine seam


class FakeDebtEngine:
    """Duck-typed DebtEngine returning one canned report; records calls."""

    calls: list[tuple[str, DebtOptions | None]] = []
    report: DebtReport
    error: Exception | None = None

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def debt_project(
        self,
        project: Any,
        app_config: Any,
        options: DebtOptions | None = None,
        freshness: Any = None,
    ) -> DebtReport:
        if FakeDebtEngine.error is not None:
            raise FakeDebtEngine.error
        FakeDebtEngine.calls.append(("project", options if options is not None else DebtOptions()))
        return FakeDebtEngine.report

    def debt_path(self, path: Path, app_config: Any, options: DebtOptions | None = None) -> DebtReport:
        if FakeDebtEngine.error is not None:
            raise FakeDebtEngine.error
        FakeDebtEngine.calls.append(("path", options if options is not None else DebtOptions()))
        return FakeDebtEngine.report


def _report(max_age_days: float | None = None, passed: bool | None = None) -> DebtReport:
    themes = (
        DebtTheme(
            label="retry / backoff",
            size=2,
            matches=(
                DebtMatch(file_path="src/net/client.py", line_start=88, marker="FIXME", snippet="retry loop ignores backoff header"),
                DebtMatch(file_path="src/net/pool.py", line_start=41, marker="TODO", snippet="same backoff handling as client.py"),
            ),
            oldest_epoch=1_700_000_000,
        ),
        DebtTheme(
            label="auth / token",
            size=2,
            matches=(DebtMatch(file_path="src/auth/session.py", line_start=12, marker="HACK", snippet="token refresh races logout"),),
            oldest_epoch=None,
        ),
    )
    return DebtReport(
        themes=themes,
        scattered=1,
        markers_found=5,
        chunks_scanned=42,
        truncated=False,
        threshold=0.8,
        max_age_days=max_age_days,
        passed=passed,
    )


@pytest.fixture()
def fake_debt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolate HOME config and swap the lazy debt/project seams for fakes."""
    home_config = tmp_path / "home" / ".config" / "simgrep"
    home_config.mkdir(parents=True)
    save_app_config(AppConfig(model="fake"), home_config / "config.toml")
    monkeypatch.setenv("HOME", str(home_config.parents[1]))

    corpus = tmp_path / "corpus"
    (corpus / "net").mkdir(parents=True)
    # Two equal-length marker clusters + one singleton, as in the e2e fixture.
    (corpus / "net" / "client.py").write_text("def get(url):\n    # FIXME retry loop ignores backoff header\n    return url\n", encoding="utf-8")
    (corpus / "net" / "pool.py").write_text("def put(url):\n    # TODO same backoff handling as client.py\n    return url\n", encoding="utf-8")
    (corpus / "notes.md").write_text("# notes\n\n<!-- TODO rotate the deploy key soon -->\n", encoding="utf-8")

    FakeDebtEngine.calls = []
    FakeDebtEngine.report = _report()
    FakeDebtEngine.error = None

    class FakeActive:
        model = "other-model"
        chunk_size = 64
        chunk_overlap = 8
        indexed_paths = (corpus,)
        root = corpus

    class FakeRuntimeFactory:
        def for_app(self, app_config: Any) -> Any:
            return object()

        def for_project(self, project: Any) -> Any:
            return object()

    import simgrep.debt_engine as debt_engine_module
    import simgrep.execution as runtime_module
    import simgrep.project as project_module

    monkeypatch.setattr(project_module, "find_active_project", lambda path: FakeActive())
    monkeypatch.setattr(runtime_module, "RuntimeFactory", FakeRuntimeFactory)
    monkeypatch.setattr(debt_engine_module, "DebtEngine", FakeDebtEngine)
    return corpus


# ----------------------------------------------------------------------- registry


def test_registry_debt_is_the_last_tool() -> None:
    tools = _rpc("tools/list", {})["result"]["tools"]
    assert len(tools) == 9  # debt: ninth and final tool
    by_name = {tool["name"]: tool for tool in tools}
    assert set(by_name) == _REGISTRY
    assert [tool["name"] for tool in tools][-1:] == ["debt"]
    description = by_name["debt"]["description"].lower()
    assert "debt" in description and "marker" in description and "theme" in description


def test_debt_schema_pins_bounds_and_all_optional_arguments() -> None:
    tools = _rpc("tools/list", {})["result"]["tools"]
    spec = next(tool for tool in tools if tool["name"] == "debt")
    schema = spec["inputSchema"]
    assert schema["type"] == "object"
    assert schema.get("required", []) == []
    props = schema["properties"]
    assert set(props) == {"path", "threshold", "min_size", "top", "max_members", "max_age_days"}
    assert props["threshold"]["minimum"] == 0.01
    assert props["threshold"]["maximum"] == 1
    assert props["min_size"]["minimum"] == 1
    assert props["top"]["minimum"] == 1
    assert props["top"]["maximum"] == 200
    assert props["max_members"]["minimum"] == 1
    assert props["max_members"]["maximum"] == 50
    assert props["max_age_days"]["minimum"] == 0.01


@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        ({"threshold": 0}, "'threshold' must be >="),
        ({"threshold": 1.5}, "'threshold' must be <="),
        ({"threshold": "loose"}, "'threshold' must be of type number"),
        ({"min_size": 0}, "'min_size' must be >="),
        ({"top": 0}, "'top' must be >="),
        ({"top": 201}, "'top' must be <="),
        ({"max_members": 51}, "'max_members' must be <="),
        ({"max_age_days": 0}, "'max_age_days' must be >="),
        ({"path": 3}, "'path' must be of type string"),
    ],
)
def test_schema_violations_are_tool_errors(arguments: dict[str, Any], fragment: str) -> None:
    """SEP-1303: input validation failures are isError results, never tracebacks."""
    response = _call_debt(arguments)
    assert response["result"]["isError"] is True
    assert fragment in _result_text(response)


# ----------------------------------------------------------------------- duality


def test_path_inside_active_project_runs_persistent(fake_debt: Path) -> None:
    _payload(_call_debt({"path": str(fake_debt)}))
    assert [mode for mode, _ in FakeDebtEngine.calls] == ["project"]


def test_path_outside_active_project_runs_ephemeral(fake_debt: Path) -> None:
    other = fake_debt.parent / "elsewhere"
    other.mkdir()
    payload = _payload(_call_debt({"path": str(other)}))
    assert [mode for mode, _ in FakeDebtEngine.calls] == ["path"]
    assert payload["themes"][1]["matches"][0]["marker"] == "HACK"


def test_no_path_uses_active_project(fake_debt: Path) -> None:
    _payload(_call_debt({}))
    assert [mode for mode, _ in FakeDebtEngine.calls] == ["project"]


def test_handler_passes_caller_tuning_to_the_engine(fake_debt: Path) -> None:
    _call_debt({"path": str(fake_debt), "threshold": 0.7, "min_size": 3, "top": 5, "max_members": 4})
    _, options = FakeDebtEngine.calls[0]
    assert options == DebtOptions(threshold=0.7, min_size=3, top=5, max_members=4)


# ----------------------------------------------------------------------- payload parity


def test_happy_path_payload_matches_pinned_cli_json_contract(fake_debt: Path) -> None:
    from simgrep.records import debt_record

    report = FakeDebtEngine.report
    payload = _payload(_call_debt({"path": str(fake_debt)}))
    # Parity rule: the MCP payload is exactly the CLI json record for the same report.
    assert payload == json.loads(json.dumps(debt_record(report)))
    assert set(payload) == _PAYLOAD_KEYS
    assert payload["scattered"] == 1
    assert payload["markers_found"] == 5
    assert payload["chunks_scanned"] == 42
    assert payload["truncated"] is False
    assert payload["threshold"] == pytest.approx(0.8)
    assert payload["max_age_days"] is None
    assert payload["passed"] is None
    assert len(payload["themes"]) == 2
    first = payload["themes"][0]
    assert set(first) == _THEME_KEYS
    assert first["label"] == "retry / backoff"
    assert first["size"] == 2
    assert first["oldest_epoch"] == 1_700_000_000
    assert len(first["matches"]) == 2
    assert set(first["matches"][0]) == _MATCH_KEYS
    assert first["matches"][0] == {
        "file_path": "src/net/client.py",
        "line_start": 88,
        "marker": "FIXME",
        "snippet": "retry loop ignores backoff header",
    }


def test_gate_arguments_flow_into_max_age_days_and_passed(fake_debt: Path) -> None:
    FakeDebtEngine.report = _report(max_age_days=90, passed=True)
    payload = _payload(_call_debt({"path": str(fake_debt), "max_age_days": 90}))
    assert payload["max_age_days"] == pytest.approx(90)
    assert payload["passed"] is True


# ----------------------------------------------------------------------- error paths


def test_missing_active_project_is_a_clean_tool_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No path and no active project mirrors the CLI guard verbatim."""
    home_config = tmp_path / "home" / ".config" / "simgrep"
    home_config.mkdir(parents=True)
    save_app_config(AppConfig(model="fake"), home_config / "config.toml")
    monkeypatch.setenv("HOME", str(home_config.parents[1]))
    import simgrep.project as project_module

    monkeypatch.setattr(project_module, "find_active_project", lambda path: None)
    response = _call_debt({})
    assert response["result"]["isError"] is True
    assert "no active project" in _result_text(response).lower()


def test_engine_debt_error_is_a_clean_tool_error_not_a_traceback(fake_debt: Path) -> None:
    FakeDebtEngine.error = DebtError("corpus exceeds the scan guard", hint="Narrow the scope (e.g. a subdirectory).")
    response = _call_debt({"path": str(fake_debt)})
    assert response["result"]["isError"] is True
    text = _result_text(response)
    assert "corpus exceeds the scan guard" in text
    assert "Narrow the scope" in text
    assert "Traceback" not in text
