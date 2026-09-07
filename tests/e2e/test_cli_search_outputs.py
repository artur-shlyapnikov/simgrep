import pathlib
from typing import Any

from .conftest import assert_clean_json_list, assert_failure_contains, assert_paths_only, assert_success, run_simgrep_command


def test_persistent_output_modes(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    project_dir = tmp_path / "persist_docs"
    docs_dir = project_dir / "docs"
    project_dir.mkdir()
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("apple banana kiwi")

    assert_success(run_simgrep_command(["init"], cwd=project_dir))
    assert_success(run_simgrep_command(["project", "add-path", str(docs_dir)], cwd=project_dir))
    assert_success(run_simgrep_command(["index", "--rebuild"], cwd=project_dir))

    json_result = run_simgrep_command(["search", "banana", "--format", "json"], cwd=project_dir)
    payload = assert_clean_json_list(json_result)
    assert payload and "path" in payload[0]

    paths_result = run_simgrep_command(["search", "banana", "--format", "paths"], cwd=project_dir)
    paths = assert_paths_only(paths_result)
    assert any("a.txt" in p for p in paths)


def test_ephemeral_json_cleanliness(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    docs_dir = tmp_path / "ephemeral_docs"
    docs_dir.mkdir()
    (docs_dir / "e.txt").write_text("ephemeral output data")

    json_result = run_simgrep_command(["search", "output data", str(docs_dir), "--format", "json"])
    payload = assert_clean_json_list(json_result)
    assert any("e.txt" in item.get("path", "") for item in payload)


def test_json_output_includes_line_fields(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    docs_dir = tmp_path / "json_line_docs"
    docs_dir.mkdir()
    (docs_dir / "e.txt").write_text("line one\nline two data\nline three")

    json_result = run_simgrep_command(["search", "line two", str(docs_dir), "--format", "json"])
    payload = assert_clean_json_list(json_result)
    assert payload
    first = payload[0]
    assert "line_start" in first
    assert "line_end" in first


def test_json_output_includes_why_when_enabled(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    docs_dir = tmp_path / "json_why_docs"
    docs_dir.mkdir()
    (docs_dir / "w.txt").write_text("alpha beta gamma")

    json_result = run_simgrep_command(["search", "alpha beta", str(docs_dir), "--format", "json", "--why"])
    payload = assert_clean_json_list(json_result)
    assert payload
    assert "why" in payload[0]
    assert isinstance(payload[0]["why"], dict)


def test_jsonl_output_includes_why_when_enabled(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    docs_dir = tmp_path / "jsonl_why_docs"
    docs_dir.mkdir()
    (docs_dir / "w.txt").write_text("delta epsilon zeta")

    result = run_simgrep_command(["search", "epsilon", str(docs_dir), "--format", "jsonl", "--why"])
    assert_success(result)
    lines = [line for line in result.stdout.strip().split("\n") if line.strip()]
    assert lines
    import json

    first = json.loads(lines[0])
    assert "why" in first
    assert isinstance(first["why"], dict)


def test_path_scope_prefers_persistent_and_ephemeral_can_override(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    project_dir = tmp_path / "scoped_project"
    docs_dir = project_dir / "scoped_docs"
    project_dir.mkdir()
    docs_dir.mkdir()
    f = docs_dir / "a.txt"
    f.write_text("scoped_persistent_term")

    assert_success(run_simgrep_command(["init"], cwd=project_dir))
    assert_success(run_simgrep_command(["project", "add-path", str(docs_dir)], cwd=project_dir))
    assert_success(run_simgrep_command(["index", "--rebuild"], cwd=project_dir))

    f.unlink()

    persisted = run_simgrep_command(["search", "scoped_persistent_term", str(docs_dir), "--freshness", "skip", "--format", "json"], cwd=project_dir)
    persisted_payload = assert_clean_json_list(persisted)
    assert persisted_payload
    assert any("a.txt" in item.get("path", "") for item in persisted_payload)

    ephemeral = run_simgrep_command(["search", "scoped_persistent_term", str(docs_dir), "--ephemeral", "--format", "json"], cwd=project_dir)
    ephemeral_payload = assert_clean_json_list(ephemeral)
    assert ephemeral_payload == []


def test_persistent_flag_fails_when_no_index_for_path(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    docs_dir = tmp_path / "persist_required_docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("hello")
    result = run_simgrep_command(["search", "hello", str(docs_dir), "--persistent", "--format", "json"])
    assert_failure_contains(result, ["persistent", "project"])


def test_no_hybrid_remains_usable_on_symbol_heavy_query(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    docs_dir = tmp_path / "symbol_docs"
    docs_dir.mkdir()
    (docs_dir / "api.py").write_text("def fetch_user_by_id(user_id: str) -> dict: return {}", encoding="utf-8")

    result = run_simgrep_command(
        [
            "search",
            "fetch_user_by_id",
            str(docs_dir),
            "--no-hybrid",
            "--format",
            "json",
            "--top",
            "3",
        ]
    )
    payload = assert_clean_json_list(result)
    assert payload
    assert any("api.py" in item.get("path", "") for item in payload)


def test_e2e_lexical_fallback_modes_off_fill_empty(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Pin the CLI-visible lexical-fallback contract using payload classification
    (why.lexical_only), never filename-vs-model assumptions about which document wins
    the semantic ranking."""
    docs_dir = tmp_path / "fallback_modes_docs"
    docs_dir.mkdir()
    (docs_dir / "aaa_semantic_first.py").write_text("stable alpha context", encoding="utf-8")
    (docs_dir / "zzz_lexical_only.md").write_text("needle needle needle", encoding="utf-8")

    base_args = [
        "search",
        "needle",
        str(docs_dir),
        "--format",
        "json",
        "--top",
        "5",
        "--candidates",
        "1",
        "--lexical-top",
        "10",
        "--lexical-weight",
        "0.5",
        "--why",
    ]

    payloads: dict[str, list[dict[str, Any]]] = {}
    for mode in ("off", "fill", "empty"):
        payloads[mode] = assert_clean_json_list(run_simgrep_command([*base_args, "--lexical-fallback", mode]))

    for mode, payload in payloads.items():
        assert payload, f"{mode}: expected at least one row"
        for row in payload:
            assert isinstance(row.get("why"), dict), f"{mode}: missing why payload"
            assert isinstance(row["why"].get("lexical_only"), bool), f"{mode}: why.lexical_only missing"

    def _lexonly(mode: str) -> list[dict[str, Any]]:
        return [row for row in payloads[mode] if row["why"]["lexical_only"]]

    # off: every surfaced row carries a semantic component.
    assert _lexonly("off") == []
    # fill: any lexical-only survivor must not outrank the weakest semantic row.
    semantic_scores = [row["score"] for row in payloads["fill"] if not row["why"]["lexical_only"]]
    if _lexonly("fill") and semantic_scores:
        assert all(row["score"] <= min(semantic_scores) for row in _lexonly("fill"))
    # empty: any lexical-only survivor scores exactly zero.
    assert all(row["score"] == 0.0 for row in _lexonly("empty"))
