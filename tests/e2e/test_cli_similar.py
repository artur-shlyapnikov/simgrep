"""End-to-end tests for the `simgrep similar` CLI command."""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from .conftest import assert_clean_json_list, assert_clean_jsonl, assert_failure_contains, assert_success, run_simgrep_command

COMMON = "def retry_request():\n    return retry(request)\n"


@pytest.fixture
def indexed_project(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> pathlib.Path:
    project_dir = tmp_path / "similar_project"
    project_dir.mkdir()
    (project_dir / "a.py").write_text("MARKER_A unique alpha\n" + COMMON, encoding="utf-8")
    (project_dir / "b.py").write_text("MARKER_B unique beta\n" + COMMON, encoding="utf-8")
    (project_dir / "c.py").write_text("MARKER_C unique gamma\n" + COMMON, encoding="utf-8")
    (project_dir / "d.py").write_text("completely different words here entirely\n", encoding="utf-8")
    assert_success(run_simgrep_command(["init"], cwd=project_dir))
    assert_success(run_simgrep_command(["project", "add-path", str(project_dir)], cwd=project_dir))
    assert_success(run_simgrep_command(["index", "--rebuild"], cwd=project_dir))
    return project_dir


class TestSimilarOutputs:
    def test_json_format_excludes_source_file(self, indexed_project: pathlib.Path) -> None:
        result = run_simgrep_command(["similar", f"@{indexed_project / 'a.py'}", "--format", "json"], cwd=indexed_project)
        payload = assert_clean_json_list(result)
        paths = [item["path"] for item in payload]
        assert any("b.py" in p for p in paths)
        assert not any("a.py" in p for p in paths)

    def test_line_range_anchor(self, indexed_project: pathlib.Path) -> None:
        anchor = f"{indexed_project / 'b.py'}:1-3"
        result = run_simgrep_command(["similar", anchor, "--format", "json"], cwd=indexed_project)
        payload = assert_clean_json_list(result)
        paths = [item["path"] for item in payload]
        assert any("a.py" in p or "c.py" in p for p in paths)
        assert not any(p.endswith("b.py") for p in paths)

    def test_stdin_anchor(self, indexed_project: pathlib.Path) -> None:
        result = run_simgrep_command(
            ["similar", "-", "--format", "json"],
            cwd=indexed_project,
            input_str="retry request retry_request\n",
        )
        payload = assert_clean_json_list(result)
        assert len(payload) > 0

    def test_grep_format_shape(self, indexed_project: pathlib.Path) -> None:
        result = run_simgrep_command(["similar", f"@{indexed_project / 'a.py'}", "--format", "grep"], cwd=indexed_project)
        assert_success(result)
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        assert lines
        for line in lines:
            assert re.match(r"^[^:]+:\d+(:[\d.]+)?: ", line), f"malformed grep line: {line}"

    def test_empty_stdin_anchor_errors(self, indexed_project: pathlib.Path) -> None:
        result = run_simgrep_command(["similar", "-", "--format", "json"], cwd=indexed_project, input_str="   \n")
        assert_failure_contains(result, ["empty"])

    def test_include_self_flag(self, indexed_project: pathlib.Path) -> None:
        result = run_simgrep_command(["similar", f"@{indexed_project / 'a.py'}", "--format", "jsonl", "--include-self"], cwd=indexed_project)
        rows = assert_clean_jsonl(result)
        assert any(row["path"].endswith("a.py") for row in rows)

    def test_single_line_anchor_excludes_anchor_file(self, indexed_project: pathlib.Path) -> None:
        result = run_simgrep_command(["similar", f"{indexed_project / 'a.py'}:1", "--format", "json"], cwd=indexed_project)
        payload = assert_clean_json_list(result)
        paths = [item["path"] for item in payload]
        assert paths
        assert not any("a.py" in p for p in paths)

    def test_file_like_line_anchor_without_file_errors_with_hint(self) -> None:
        result = run_simgrep_command(["similar", "greet.py:6", "--format", "json"])
        assert_failure_contains(result, ["not found", "greet.py:6"])


class TestContrastiveCli:
    def test_unlike_changes_scores(self, indexed_project: pathlib.Path) -> None:
        plain = run_simgrep_command(["similar", f"@{indexed_project / 'b.py'}", "--format", "json"], cwd=indexed_project)
        contrastive = run_simgrep_command(
            ["similar", f"@{indexed_project / 'b.py'}", "--unlike", f"@{indexed_project / 'a.py'}", "--format", "json"],
            cwd=indexed_project,
        )
        plain_rows = json.loads(plain.stdout)
        contrastive_rows = json.loads(contrastive.stdout)
        assert plain_rows and contrastive_rows
        assert max(row["score"] for row in contrastive_rows) < max(row["score"] for row in plain_rows)

    def test_unlike_weight_out_of_bounds_errors(self, indexed_project: pathlib.Path) -> None:
        result = run_simgrep_command(
            [
                "similar",
                f"@{indexed_project / 'b.py'}",
                "--unlike",
                f"{indexed_project / 'a.py'}:1-1",
                "--unlike-weight",
                "1.5",
                "--format",
                "json",
            ],
            cwd=indexed_project,
        )
        assert_failure_contains(result, ["unlike-weight"])

    def test_unlike_adds_why_keys(self, indexed_project: pathlib.Path) -> None:
        result = run_simgrep_command(
            [
                "similar",
                f"@{indexed_project / 'b.py'}",
                "--unlike",
                f"@{indexed_project / 'a.py'}",
                "--why",
                "--format",
                "json",
            ],
            cwd=indexed_project,
        )
        rows = json.loads(result.stdout)
        assert rows
        assert "semantic_like" in rows[0]["why"]
        assert "semantic_unlike" in rows[0]["why"]

    def test_without_unlike_why_matches_search_shape(self, indexed_project: pathlib.Path) -> None:
        search_result = run_simgrep_command(["search", "retry request", "--why", "--format", "json"], cwd=indexed_project)
        similar_result = run_simgrep_command(["similar", f"@{indexed_project / 'b.py'}", "--why", "--format", "json"], cwd=indexed_project)
        search_rows = json.loads(search_result.stdout)
        similar_rows = json.loads(similar_result.stdout)
        assert search_rows and similar_rows
        assert set(similar_rows[0]["why"]) == set(search_rows[0]["why"])


class TestAnchorDecodingCli:
    def test_latin1_anchor_file_clean_error(self, indexed_project: pathlib.Path) -> None:
        bad = indexed_project / "latin1_ref.py"
        bad.write_bytes(b"caf\xe9 retry request\n")
        result = run_simgrep_command(["similar", f"@{bad}", "--format", "json"], cwd=indexed_project)
        assert_failure_contains(result, ["utf-8"])


class TestUnlikeStdinCli:
    def test_unlike_reads_stdin_when_like_is_file(self, indexed_project: pathlib.Path) -> None:
        result = run_simgrep_command(
            ["similar", f"@{indexed_project / 'b.py'}", "--unlike", "-", "--format", "json"],
            cwd=indexed_project,
            input_str=COMMON,
        )
        rows = assert_clean_json_list(result)
        assert rows
