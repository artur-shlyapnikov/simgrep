"""Contract tests: all CLI commands from README use-cases.

These tests verify the exact commands documented in README.md work as described.
They also document the actual default behavior for key features like `why` and `hybrid`.
"""

import json
import pathlib
import re

import pytest

from .conftest import (
    assert_clean_json_list,
    assert_paths_only,
    assert_success,
    run_simgrep_command,
)


class TestInitCommand:
    def test_init_creates_project_config(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        project_dir = tmp_path / "init_project"
        project_dir.mkdir()

        result = run_simgrep_command(["init"], cwd=project_dir)
        assert_success(result)
        assert "Initialized simgrep project" in result.stdout
        assert (project_dir / ".simgrep" / "project.toml").exists()

    def test_init_twice_without_yes_fails(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        project_dir = tmp_path / "init_twice"
        project_dir.mkdir()

        assert_success(run_simgrep_command(["init"], cwd=project_dir))
        second = run_simgrep_command(["init"], cwd=project_dir)
        assert second.exit_code != 0

    def test_init_with_yes_overwrites(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        project_dir = tmp_path / "init_overwrite"
        project_dir.mkdir()

        assert_success(run_simgrep_command(["init"], cwd=project_dir))
        result = run_simgrep_command(["init", "--yes"], cwd=project_dir)
        assert_success(result)


class TestIndexCommand:
    @pytest.fixture
    def indexed_project(self, tmp_path: pathlib.Path) -> pathlib.Path:
        project_dir = tmp_path / "index_project"
        docs_dir = project_dir / "docs"
        project_dir.mkdir()
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("indexable content", encoding="utf-8")

        assert_success(run_simgrep_command(["init"], cwd=project_dir))
        assert_success(run_simgrep_command(["project", "add-path", str(docs_dir)], cwd=project_dir))
        assert_success(run_simgrep_command(["index", "--rebuild"], cwd=project_dir))
        return project_dir

    def test_index_reports_stats(self, temp_simgrep_home: pathlib.Path, indexed_project: pathlib.Path) -> None:
        result = run_simgrep_command(["index"], cwd=indexed_project)
        assert_success(result)
        assert "file(s)" in result.stdout
        assert "chunk(s)" in result.stdout

    def test_index_dry_run(self, temp_simgrep_home: pathlib.Path, indexed_project: pathlib.Path) -> None:
        result = run_simgrep_command(["index", "--dry-run"], cwd=indexed_project)
        assert_success(result)
        assert "Would index" in result.stdout


class TestSearchEphemeral:
    def test_search_ephemeral_with_path(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "ephemeral_path"
        docs_dir.mkdir()
        (docs_dir / "src.txt").write_text("async function examples here", encoding="utf-8")

        result = run_simgrep_command(["search", "async function examples", str(docs_dir), "--format", "json"])
        assert_success(result)
        payload = assert_clean_json_list(result)
        assert len(payload) > 0

    def test_search_ephemeral_preserves_path(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "eph_preserve"
        docs_dir.mkdir()
        (docs_dir / "b.txt").write_text("hello world content", encoding="utf-8")

        result = run_simgrep_command(["search", "hello world", str(docs_dir), "--format", "json"])
        assert_success(result)
        payload = assert_clean_json_list(result)
        assert any("b.txt" in item["path"] for item in payload)


class TestProjectAddPath:
    def test_project_add_path(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        project_dir = tmp_path / "add_path_project"
        docs_dir = project_dir / "docs"
        project_dir.mkdir()
        docs_dir.mkdir()
        (docs_dir / "readme.txt").write_text("docs content", encoding="utf-8")

        assert_success(run_simgrep_command(["init"], cwd=project_dir))
        result = run_simgrep_command(["project", "add-path", str(docs_dir)], cwd=project_dir)
        assert_success(result)
        assert "Indexed paths:" in result.stdout


class TestProjectRemovePath:
    def test_project_remove_path(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        project_dir = tmp_path / "remove_path_project"
        docs_dir = project_dir / "docs"
        project_dir.mkdir()
        docs_dir.mkdir()
        (docs_dir / "old.txt").write_text("to be removed", encoding="utf-8")

        assert_success(run_simgrep_command(["init"], cwd=project_dir))
        assert_success(run_simgrep_command(["project", "add-path", str(docs_dir)], cwd=project_dir))
        result = run_simgrep_command(["project", "remove-path", str(docs_dir)], cwd=project_dir)
        assert_success(result)
        assert "Indexed paths:" in result.stdout


class TestStatusCommand:
    @pytest.fixture
    def status_project(self, tmp_path: pathlib.Path) -> pathlib.Path:
        project_dir = tmp_path / "status_project"
        docs_dir = project_dir / "docs"
        project_dir.mkdir()
        docs_dir.mkdir()
        (docs_dir / "status.txt").write_text("status content", encoding="utf-8")

        assert_success(run_simgrep_command(["init"], cwd=project_dir))
        assert_success(run_simgrep_command(["project", "add-path", str(docs_dir)], cwd=project_dir))
        assert_success(run_simgrep_command(["index", "--rebuild"], cwd=project_dir))
        return project_dir

    def test_status_shows_index_info(self, temp_simgrep_home: pathlib.Path, status_project: pathlib.Path) -> None:
        result = run_simgrep_command(["status"], cwd=status_project)
        assert_success(result)
        assert "file(s)" in result.stdout
        assert "chunk(s)" in result.stdout

    def test_status_no_index(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        project_dir = tmp_path / "no_index_project"
        project_dir.mkdir()
        (project_dir / "a.txt").write_text("content", encoding="utf-8")

        assert_success(run_simgrep_command(["init"], cwd=project_dir))
        result = run_simgrep_command(["status"], cwd=project_dir)
        assert_success(result)
        assert "no index" in result.stdout


class TestConfigGetSet:
    def test_config_get_default(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        project_dir = tmp_path / "config_get_project"
        project_dir.mkdir()
        (project_dir / "a.txt").write_text("content", encoding="utf-8")

        assert_success(run_simgrep_command(["init"], cwd=project_dir))
        result = run_simgrep_command(["config", "get", "lexical_top"], cwd=project_dir)
        assert_success(result)
        assert "50" in result.stdout

    def test_config_set_and_get(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        project_dir = tmp_path / "config_set_project"
        project_dir.mkdir()
        (project_dir / "a.txt").write_text("content", encoding="utf-8")

        assert_success(run_simgrep_command(["init"], cwd=project_dir))
        set_result = run_simgrep_command(["config", "set", "lexical_top", "25"], cwd=project_dir)
        assert_success(set_result)
        assert "25" in set_result.stdout

        get_result = run_simgrep_command(["config", "get", "lexical_top"], cwd=project_dir)
        assert_success(get_result)
        assert "25" in get_result.stdout

    def test_config_list_shows_values(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        project_dir = tmp_path / "config_list_project"
        project_dir.mkdir()
        (project_dir / "a.txt").write_text("content", encoding="utf-8")

        assert_success(run_simgrep_command(["init"], cwd=project_dir))
        result = run_simgrep_command(["config", "list"], cwd=project_dir)
        assert_success(result)
        assert "lexical_top" in result.stdout


class TestModelsStatus:
    def test_models_status_nonexistent(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        del tmp_path
        result = run_simgrep_command(["models", "status", "definitely/not-a-real-model"])
        assert_success(result)
        assert "not cached" in result.stdout.lower() or "cached" in result.stdout.lower()


class TestResetCommand:
    @pytest.fixture
    def reset_project(self, tmp_path: pathlib.Path) -> pathlib.Path:
        project_dir = tmp_path / "reset_project"
        docs_dir = project_dir / "docs"
        project_dir.mkdir()
        docs_dir.mkdir()
        (docs_dir / "reset.txt").write_text("reset content", encoding="utf-8")

        assert_success(run_simgrep_command(["init"], cwd=project_dir))
        assert_success(run_simgrep_command(["project", "add-path", str(docs_dir)], cwd=project_dir))
        assert_success(run_simgrep_command(["index", "--rebuild"], cwd=project_dir))
        return project_dir

    def test_reset_deletes_artifacts(self, temp_simgrep_home: pathlib.Path, reset_project: pathlib.Path) -> None:
        assert (reset_project / ".simgrep" / "metadata.duckdb").exists()
        assert (reset_project / ".simgrep" / "vectors.usearch").exists()

        result = run_simgrep_command(["reset", "--yes"], cwd=reset_project)
        assert_success(result)
        assert "Reset" in result.stdout

        assert not (reset_project / ".simgrep" / "vectors.usearch").exists()


class TestWhyDefaultBehavior:
    """Document actual default: why is NOT shown by default (code vs docs discrepancy).

    The code has: why: bool = typer.Option(False, "--why/--no-why")
    But README claims: "Per-result ranking explanation is shown by default (`--no-why` to hide)"

    These tests verify the ACTUAL behavior (False by default) matches the CODE not the docs.
    Either the docs should be updated or the code should be changed to match docs.
    """

    def test_why_absent_by_default_json(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "why_default"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("why test content alpha", encoding="utf-8")

        result = run_simgrep_command(["search", "alpha", str(docs_dir), "--format", "json"])
        payload = assert_clean_json_list(result)
        assert payload
        assert "why" not in payload[0]

    def test_why_absent_by_default_jsonl(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "why_default_jsonl"
        docs_dir.mkdir()
        (docs_dir / "b.txt").write_text("beta why content", encoding="utf-8")

        result = run_simgrep_command(["search", "beta", str(docs_dir), "--format", "jsonl"])
        assert_success(result)
        lines = [line for line in result.stdout.strip().split("\n") if line.strip()]
        assert lines
        first = json.loads(lines[0])
        assert "why" not in first

    def test_why_flag_enables_why_output(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "why_enabled"
        docs_dir.mkdir()
        (docs_dir / "c.txt").write_text("gamma content for why", encoding="utf-8")

        result = run_simgrep_command(["search", "gamma", str(docs_dir), "--why", "--format", "json"])
        payload = assert_clean_json_list(result)
        assert payload
        assert "why" in payload[0]

    def test_no_why_explicit_flag(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "no_why_explicit"
        docs_dir.mkdir()
        (docs_dir / "d.txt").write_text("delta content", encoding="utf-8")

        result = run_simgrep_command(["search", "delta", str(docs_dir), "--no-why", "--format", "json"])
        payload = assert_clean_json_list(result)
        assert payload
        assert "why" not in payload[0]


class TestMachineFormatCleanliness:
    """Machine formats (json, jsonl, paths, count, grep) must produce clean stdout with no stderr."""

    def test_json_clean_stdout_no_stderr(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "fmt_json"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("machine format test", encoding="utf-8")

        result = run_simgrep_command(["search", "machine", str(docs_dir), "--format", "json"])
        assert_success(result)
        assert result.stderr == ""
        payload = json.loads(result.stdout)
        assert isinstance(payload, list)

    def test_jsonl_clean_stdout_no_stderr(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "fmt_jsonl"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("jsonl format content", encoding="utf-8")

        result = run_simgrep_command(["search", "jsonl", str(docs_dir), "--format", "jsonl"])
        assert_success(result)
        assert result.stderr == ""
        for line in result.stdout.splitlines():
            if line.strip():
                json.loads(line)

    def test_paths_clean_stdout_no_stderr(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "fmt_paths"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("paths format content", encoding="utf-8")

        result = run_simgrep_command(["search", "paths", str(docs_dir), "--format", "paths"])
        assert_success(result)
        assert result.stderr == ""
        paths = assert_paths_only(result)
        assert paths

    def test_count_clean_stdout_no_stderr(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "fmt_count"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("count format content", encoding="utf-8")

        result = run_simgrep_command(["search", "count", str(docs_dir), "--format", "count"])
        assert_success(result)
        assert result.stderr == ""
        lines = result.stdout.strip().split("\n")
        assert len(lines) == 1
        assert lines[0].isdigit()

    def test_grep_clean_stdout_no_stderr(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "fmt_grep"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("grep format line one\nline two match\nline three", encoding="utf-8")

        result = run_simgrep_command(["search", "match", str(docs_dir), "--format", "grep"])
        assert_success(result)
        assert result.stderr == ""
        lines = result.stdout.strip().split("\n")
        for line in lines:
            assert re.match(r"^[^:]+:\d+(:[\d.]+)?: ", line), f"malformed grep line: {line}"


class TestReadmeCommandsContract:
    """End-to-end contract test for all README-documented commands in sequence."""

    def test_full_workflow_init_index_search_add_path_remove_path_status(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        project_dir = tmp_path / "full_contract_project"
        docs_dir = project_dir / "docs"
        backend_dir = project_dir / "backend"
        project_dir.mkdir()
        docs_dir.mkdir()
        backend_dir.mkdir()

        (docs_dir / "intro.txt").write_text("simgrep project docs content", encoding="utf-8")
        (backend_dir / "api.py").write_text("backend api code content", encoding="utf-8")

        # 1. simgrep init
        r_init = run_simgrep_command(["init"], cwd=project_dir)
        assert_success(r_init)
        assert "Initialized simgrep project" in r_init.stdout

        # 2. simgrep project add-path docs
        r_add_docs = run_simgrep_command(["project", "add-path", str(docs_dir)], cwd=project_dir)
        assert_success(r_add_docs)

        # 3. simgrep project add-path backend
        r_add_backend = run_simgrep_command(["project", "add-path", str(backend_dir)], cwd=project_dir)
        assert_success(r_add_backend)

        # 4. simgrep index
        r_index = run_simgrep_command(["index", "--rebuild"], cwd=project_dir)
        assert_success(r_index)
        assert "Indexed" in r_index.stdout

        # 5. simgrep search (persistent, no path)
        r_search = run_simgrep_command(["search", "docs content", "--format", "json"], cwd=project_dir)
        assert_success(r_search)
        payload = assert_clean_json_list(r_search)
        assert len(payload) > 0

        # 6. simgrep search ... --ephemeral
        r_eph = run_simgrep_command(["search", "api code", str(backend_dir), "--ephemeral", "--format", "json"])
        assert_success(r_eph)

        # 7. simgrep project remove-path docs
        r_remove = run_simgrep_command(["project", "remove-path", str(docs_dir)], cwd=project_dir)
        assert_success(r_remove)

        # 8. simgrep status
        r_status = run_simgrep_command(["status"], cwd=project_dir)
        assert_success(r_status)

        # 9. simgrep config get
        r_config_get = run_simgrep_command(["config", "get", "lexical_top"], cwd=project_dir)
        assert_success(r_config_get)

        # 10. simgrep config set
        r_config_set = run_simgrep_command(["config", "set", "lexical_top", "30"], cwd=project_dir)
        assert_success(r_config_set)

        # 11. simgrep models status
        r_models = run_simgrep_command(["models", "status"], cwd=project_dir)
        assert_success(r_models)

        # 12. simgrep reset --yes
        r_reset = run_simgrep_command(["reset", "--yes"], cwd=project_dir)
        assert_success(r_reset)
        assert "Reset" in r_reset.stdout

        # After reset, artifacts should be gone
        assert not (project_dir / ".simgrep" / "metadata.duckdb").exists()
        assert not (project_dir / ".simgrep" / "vectors.usearch").exists()

        # 13. simgrep mcp (empty stdin -> immediate clean EOF exit)
        r_mcp = run_simgrep_command(["mcp"], cwd=project_dir, input_str="")
        assert_success(r_mcp)
        assert r_mcp.stdout == ""


class TestRelativePathsDefault:
    """Default behavior: relative paths are shown (not absolute)."""

    def test_relative_paths_by_default_json(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "rel_paths_default"
        docs_dir.mkdir()
        (docs_dir / "rel.txt").write_text("relative path test", encoding="utf-8")

        result = run_simgrep_command(["search", "relative", str(docs_dir), "--format", "json"])
        payload = assert_clean_json_list(result)
        assert payload
        for item in payload:
            path = item["path"]
            assert not path.startswith("/"), f"Expected relative path, got: {path}"

    def test_absolute_paths_flag_overrides(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "abs_paths_override"
        docs_dir.mkdir()
        (docs_dir / "abs.txt").write_text("absolute path test", encoding="utf-8")

        result = run_simgrep_command(["search", "absolute", str(docs_dir), "--absolute-paths", "--format", "json"])
        payload = assert_clean_json_list(result)
        assert payload
        for item in payload:
            path = item["path"]
            assert path.startswith("/") or (len(path) > 1 and path[1] == ":"), f"Expected absolute path, got: {path}"
