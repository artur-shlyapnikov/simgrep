import pathlib
import re

import pytest

from .conftest import assert_success, run_simgrep_command

_CORE_COMMAND_DESCRIPTIONS = {
    "init": "Create a simgrep project",
    "index": "persistent index",
    "search": "Semantic hybrid search",
    "similar": "semantically similar",
    "status": "index state",
    "repl": "Interactive semantic query loop",
    "doctor": "Sanity-check",
    "reset": "index artifacts",
}


def _make_docs(tmp_path: pathlib.Path, name: str = "docs") -> pathlib.Path:
    docs_dir = tmp_path / name
    docs_dir.mkdir()
    (docs_dir / "note.txt").write_text("the launch code is alpha-nine", encoding="utf-8")
    return docs_dir


class TestHelpDescriptions:
    @pytest.mark.parametrize(("command", "fragment"), sorted(_CORE_COMMAND_DESCRIPTIONS.items()))
    def test_core_command_help_is_descriptive(self, temp_simgrep_home: pathlib.Path, command: str, fragment: str) -> None:
        result = run_simgrep_command([command, "--help"])
        assert_success(result)
        assert fragment in result.stdout

    def test_top_level_help_lists_command_descriptions(self, temp_simgrep_home: pathlib.Path) -> None:
        result = run_simgrep_command(["--help"])
        assert_success(result)
        for fragment in _CORE_COMMAND_DESCRIPTIONS.values():
            assert fragment in result.stdout


class TestIndexProgressFeedback:
    def test_ephemeral_search_reports_progress_on_stderr_for_human_format(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = _make_docs(tmp_path)
        result = run_simgrep_command(["search", "launch code", str(docs_dir)])
        assert_success(result)
        assert "simgrep:" in result.stderr
        assert "Indexing" in result.stderr
        assert "note.txt" in result.stdout

    @pytest.mark.parametrize("fmt", ["json", "jsonl", "paths", "count", "grep"])
    def test_machine_formats_stay_silent_while_indexing(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path, fmt: str) -> None:
        docs_dir = _make_docs(tmp_path, f"docs_{fmt}")
        result = run_simgrep_command(["search", "launch code", str(docs_dir), "--format", fmt])
        assert_success(result)
        assert result.stderr == ""

    def test_similar_reports_progress_for_human_format(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = _make_docs(tmp_path)
        result = run_simgrep_command(["similar", "alpha nine launch code", str(docs_dir)])
        assert_success(result)
        assert "simgrep:" in result.stderr


class TestNoResultsGuidance:
    def test_no_matches_message_is_clean(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "docs_nomatch"
        docs_dir.mkdir()
        (docs_dir / "test.txt").write_text("hello world example", encoding="utf-8")
        result = run_simgrep_command(["search", "hello", str(docs_dir), "--min-score", "0.99"])
        assert_success(result)
        assert "No matches after filters." in result.stdout


class TestFirstRunGuidance:
    def test_init_suggests_next_steps(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        project_dir = tmp_path / "init_guidance_project"
        project_dir.mkdir()
        result = run_simgrep_command(["init"], cwd=project_dir)
        assert_success(result)
        assert "Initialized simgrep project" in result.stdout
        assert "Next:" in result.stdout
        assert "simgrep index" in result.stdout

    def test_status_without_index_suggests_index(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        project_dir = tmp_path / "status_hint_project"
        project_dir.mkdir()
        (project_dir / "a.txt").write_text("content", encoding="utf-8")
        assert_success(run_simgrep_command(["init"], cwd=project_dir))
        result = run_simgrep_command(["status"], cwd=project_dir)
        assert "no index" in result.stdout
        assert "simgrep index" in result.stdout

        import re

        project_dir = tmp_path / "index_timing_project"
        project_dir.mkdir()
        (project_dir / "a.txt").write_text("content", encoding="utf-8")
        assert_success(run_simgrep_command(["init"], cwd=project_dir))
        assert_success(run_simgrep_command(["project", "add-path", str(project_dir)], cwd=project_dir))
        result = run_simgrep_command(["index"], cwd=project_dir)
        assert_success(result)
        assert re.search(r"Indexed \d+ file\(s\), \d+ chunk\(s\) in \d+\.\d+s\.", result.stdout)


class TestDoctorDiagnostics:
    def test_doctor_reports_version_model_cache_and_index(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        project_dir = tmp_path / "doctor_rich_project"
        project_dir.mkdir()
        (project_dir / "a.txt").write_text("content", encoding="utf-8")
        assert_success(run_simgrep_command(["init"], cwd=project_dir))
        assert_success(run_simgrep_command(["project", "add-path", str(project_dir)], cwd=project_dir))
        assert_success(run_simgrep_command(["index"], cwd=project_dir))

        result = run_simgrep_command(["doctor"], cwd=project_dir)
        assert_success(result)
        assert "simgrep:" in result.stdout
        assert "config: ok" in result.stdout
        assert "not cached" in result.stdout
        assert re.search(r"index: \d+ file\(s\), \d+ chunk\(s\)", result.stdout)

    def test_doctor_without_project_still_succeeds(self, temp_simgrep_home: pathlib.Path) -> None:
        result = run_simgrep_command(["doctor"])
        assert_success(result)
        assert "project: none" in result.stdout


class TestSubcommandHelp:
    @pytest.mark.parametrize(
        ("args", "fragment"),
        [
            (["project", "add-path"], "Register a directory"),
            (["project", "remove-path"], "Drop a directory"),
            (["project", "info"], "indexed paths"),
            (["models", "status"], "local cache"),
            (["models", "cache"], "offline use"),
            (["config", "list"], "config key"),
            (["config", "get"], "config value"),
            (["config", "set"], "Persist one config value"),
        ],
    )
    def test_subcommand_help_is_descriptive(self, temp_simgrep_home: pathlib.Path, args: list[str], fragment: str) -> None:
        result = run_simgrep_command([*args, "--help"])
        assert_success(result)
        assert fragment in result.stdout

    def test_search_display_options_have_help_and_clean_metavars(self, temp_simgrep_home: pathlib.Path) -> None:
        result = run_simgrep_command(["search", "--help"])
        assert_success(result)
        for fragment in (
            "Context lines around each hit",
            "Truncate snippets",
            "Stale index handling",
            "Lexical-only hits",
            "Diversify results",
            "Show match scores",
            "Show line ranges",
            "per-hit scoring breakdown",
        ):
            assert fragment in result.stdout, f"missing help fragment: {fragment!r}"
        assert "FORMAT" in result.stdout  # enum metavar no longer blows the layout
        assert "[rich|compact" not in result.stdout  # no truncated mid-word choice lists
        assert "chec\n" not in result.stdout and " k]" not in result.stdout


class TestReplFeedback:
    def test_repl_shows_hit_count_and_timing(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        project_dir = tmp_path / "repl_feedback_project"
        project_dir.mkdir()
        (project_dir / "a.txt").write_text("the launch code is alpha-nine", encoding="utf-8")
        assert_success(run_simgrep_command(["init"], cwd=project_dir))
        assert_success(run_simgrep_command(["project", "add-path", str(project_dir)], cwd=project_dir))
        assert_success(run_simgrep_command(["index"], cwd=project_dir))

        result = run_simgrep_command(["repl"], cwd=project_dir, input_str="alpha nine\n\n")
        assert_success(result)
        assert "hit(s) in" in result.stdout

    def test_repl_eof_exits_cleanly(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        project_dir = tmp_path / "repl_eof_project"
        project_dir.mkdir()
        (project_dir / "a.txt").write_text("content", encoding="utf-8")
        assert_success(run_simgrep_command(["init"], cwd=project_dir))
        assert_success(run_simgrep_command(["project", "add-path", str(project_dir)], cwd=project_dir))
        assert_success(run_simgrep_command(["index"], cwd=project_dir))

        result = run_simgrep_command(["repl"], cwd=project_dir, input_str="")
        assert_success(result)
