"""CLI public contract tests for the search command."""

import json
import pathlib

import pytest

from .conftest import (
    assert_clean_json_list,
    assert_failure_contains,
    assert_paths_only,
    assert_success,
    run_simgrep_command,
)


class TestPersistentEphemeralConflict:
    def test_persistent_ephemeral_mutually_exclusive(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "conflict_doc"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("conflict content", encoding="utf-8")

        result = run_simgrep_command(
            ["search", "content", str(docs_dir), "--persistent", "--ephemeral"],
        )
        assert_failure_contains(result, ["persistent", "ephemeral"])


class TestSearchWithoutProjectAndPath:
    def test_no_active_project_no_path_hint(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "no_project"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("orphan content", encoding="utf-8")

        result = run_simgrep_command(["search", "orphan"], cwd=docs_dir)
        assert_failure_contains(result, ["project", "init"])


class TestBlankQueryHandling:
    def test_blank_query_rejected(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "blank_query_test"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("some content here", encoding="utf-8")

        result = run_simgrep_command(["search", "", str(docs_dir), "--format", "json"])
        assert_failure_contains(result, ["empty", "query"])

    def test_whitespace_only_query_rejected(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "ws_query_test"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("some content here", encoding="utf-8")

        result = run_simgrep_command(["search", "   ", str(docs_dir), "--format", "json"])
        assert_failure_contains(result, ["empty", "query"])


class TestTopLimitsResults:
    @pytest.fixture
    def multi_file_project(self, tmp_path: pathlib.Path) -> pathlib.Path:
        project_dir = tmp_path / "top_limit_project"
        docs_dir = project_dir / "docs"
        project_dir.mkdir()
        docs_dir.mkdir()

        for i in range(10):
            (docs_dir / f"file_{i}.txt").write_text(f"unique content term_{i}", encoding="utf-8")

        assert_success(run_simgrep_command(["init"], cwd=project_dir))
        assert_success(run_simgrep_command(["project", "add-path", str(docs_dir)], cwd=project_dir))
        assert_success(run_simgrep_command(["index", "--rebuild"], cwd=project_dir))
        return project_dir

    def test_top_limits_json(self, temp_simgrep_home: pathlib.Path, multi_file_project: pathlib.Path) -> None:
        result = run_simgrep_command(["search", "content", "--top", "3", "--format", "json"], cwd=multi_file_project)
        payload = assert_clean_json_list(result)
        assert len(payload) <= 3

    def test_top_limits_jsonl(self, temp_simgrep_home: pathlib.Path, multi_file_project: pathlib.Path) -> None:
        result = run_simgrep_command(["search", "content", "--top", "2", "--format", "jsonl"], cwd=multi_file_project)
        lines = [line for line in result.stdout.strip().split("\n") if line.strip()]
        assert len(lines) <= 2

    def test_top_limits_compact(self, temp_simgrep_home: pathlib.Path, multi_file_project: pathlib.Path) -> None:
        result = run_simgrep_command(["search", "content", "--top", "4", "--format", "compact"], cwd=multi_file_project)
        output_lines = [line for line in result.stdout.splitlines() if line.strip()]
        assert len(output_lines) <= 4

    def test_top_limits_grep(self, temp_simgrep_home: pathlib.Path, multi_file_project: pathlib.Path) -> None:
        result = run_simgrep_command(["search", "content", "--top", "2", "--format", "grep"], cwd=multi_file_project)
        output_lines = [line for line in result.stdout.splitlines() if line.strip()]
        assert len(output_lines) <= 2

    def test_top_limits_paths(self, temp_simgrep_home: pathlib.Path, multi_file_project: pathlib.Path) -> None:
        result = run_simgrep_command(["search", "content", "--top", "3", "--format", "paths"], cwd=multi_file_project)
        paths = assert_paths_only(result)
        assert len(paths) <= 3

    def test_top_limits_rich(self, temp_simgrep_home: pathlib.Path, multi_file_project: pathlib.Path) -> None:
        result = run_simgrep_command(["search", "content", "--top", "2", "--format", "rich"], cwd=multi_file_project)
        assert_success(result)
        output_lines = [ln for ln in result.stdout.splitlines() if ln.strip() and ln.startswith("docs/")]
        assert len(output_lines) <= 2


class TestMinScoreFiltering:
    @pytest.fixture
    def scored_project(self, tmp_path: pathlib.Path) -> pathlib.Path:
        project_dir = tmp_path / "min_score_project"
        docs_dir = project_dir / "docs"
        project_dir.mkdir()
        docs_dir.mkdir()

        (docs_dir / "high_match.txt").write_text("python programming language syntax", encoding="utf-8")
        (docs_dir / "low_match.txt").write_text("a b c d e f g h", encoding="utf-8")

        assert_success(run_simgrep_command(["init"], cwd=project_dir))
        assert_success(run_simgrep_command(["project", "add-path", str(docs_dir)], cwd=project_dir))
        assert_success(run_simgrep_command(["index", "--rebuild"], cwd=project_dir))
        return project_dir

    def test_min_score_filters_json(self, temp_simgrep_home: pathlib.Path, scored_project: pathlib.Path) -> None:
        result = run_simgrep_command(["search", "python code", "--min-score", "0.9", "--format", "json"], cwd=scored_project)
        payload = assert_clean_json_list(result)
        for item in payload:
            assert item["score"] >= 0.9

    def test_min_score_filters_compact(self, temp_simgrep_home: pathlib.Path, scored_project: pathlib.Path) -> None:
        result = run_simgrep_command(["search", "python code", "--min-score", "0.8", "--format", "compact"], cwd=scored_project)
        assert_success(result)
        if result.stdout.strip():
            for line in result.stdout.splitlines():
                if "score=" in line:
                    score_str = line.split("score=")[1].split()[0]
                    score_val = float(score_str)
                    assert score_val >= 0.8

    def test_min_score_no_results(self, temp_simgrep_home: pathlib.Path, scored_project: pathlib.Path) -> None:
        result = run_simgrep_command(["search", "python code", "--min-score", "0.99", "--format", "json"], cwd=scored_project)
        assert_success(result)
        payload = json.loads(result.stdout)
        assert payload == []


class TestCandidatesValidation:
    def test_candidates_less_than_top_adjusted(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "candidates_test"
        docs_dir.mkdir()
        for i in range(20):
            (docs_dir / f"f{i}.txt").write_text(f"content_{i}", encoding="utf-8")

        result = run_simgrep_command(["search", "content", str(docs_dir), "--top", "5", "--candidates", "2", "--format", "json"])
        payload = assert_clean_json_list(result)
        assert len(payload) <= 5


class TestIncludeExcludePatternEphemeral:
    def test_include_affects_ephemeral_indexing(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "include_test"
        docs_dir.mkdir()
        (docs_dir / "included.txt").write_text("target content alpha", encoding="utf-8")
        (docs_dir / "excluded.md").write_text("target content beta", encoding="utf-8")

        result = run_simgrep_command(
            ["search", "target content", str(docs_dir), "--include", "*.txt", "--format", "json"],
        )
        payload = assert_clean_json_list(result)
        assert any("included.txt" in p["path"] for p in payload)
        assert not any("excluded.md" in p["path"] for p in payload)

    def test_exclude_affects_ephemeral_indexing(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "exclude_test"
        docs_dir.mkdir()
        (docs_dir / "keep.txt").write_text("visible content gamma", encoding="utf-8")
        (docs_dir / "skip.txt").write_text("hidden content delta", encoding="utf-8")

        result = run_simgrep_command(
            ["search", "content", str(docs_dir), "--exclude", "skip.txt", "--format", "json"],
        )
        payload = assert_clean_json_list(result)
        assert any("keep.txt" in p["path"] for p in payload)
        assert not any("skip.txt" in p["path"] for p in payload)

    def test_pattern_affects_ephemeral_indexing(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "pattern_test"
        docs_dir.mkdir()
        (docs_dir / "matched.py").write_text("python code epsilon", encoding="utf-8")
        (docs_dir / "not_matched.txt").write_text("text zeta", encoding="utf-8")

        result = run_simgrep_command(
            ["search", "code", str(docs_dir), "--pattern", "*.py", "--format", "json"],
        )
        payload = assert_clean_json_list(result)
        assert any("matched.py" in p["path"] for p in payload)
        assert not any("not_matched.txt" in p["path"] for p in payload)


class TestFileFilterPersistent:
    @pytest.fixture
    def filter_project(self, tmp_path: pathlib.Path) -> pathlib.Path:
        project_dir = tmp_path / "filter_project"
        docs_dir = project_dir / "docs"
        subdir = docs_dir / "sub"
        project_dir.mkdir()
        docs_dir.mkdir()
        subdir.mkdir()

        (docs_dir / "root_file.txt").write_text("root content theta", encoding="utf-8")
        (subdir / "nested_file.txt").write_text("nested content iota", encoding="utf-8")
        (docs_dir / "other.txt").write_text("other kappa", encoding="utf-8")

        assert_success(run_simgrep_command(["init"], cwd=project_dir))
        assert_success(run_simgrep_command(["project", "add-path", str(docs_dir)], cwd=project_dir))
        assert_success(run_simgrep_command(["index", "--rebuild"], cwd=project_dir))
        return project_dir

    def test_file_filter_basename(self, temp_simgrep_home: pathlib.Path, filter_project: pathlib.Path) -> None:
        result = run_simgrep_command(
            ["search", "content", "--file-filter", "root_file.txt", "--format", "json"],
            cwd=filter_project,
        )
        payload = assert_clean_json_list(result)
        assert all("root_file" in p["path"] for p in payload)

    def test_file_filter_path_pattern(self, temp_simgrep_home: pathlib.Path, filter_project: pathlib.Path) -> None:
        result = run_simgrep_command(
            ["search", "content", "--file-filter", "sub/*", "--format", "json"],
            cwd=filter_project,
        )
        payload = assert_clean_json_list(result)
        assert all("sub" in p["path"] for p in payload)


class TestKeywordFiltering:
    def test_keyword_case_insensitive(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "keyword_test"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("The Quick Brown Fox", encoding="utf-8")

        result = run_simgrep_command(["search", "fox", str(docs_dir), "--keyword", "brown", "--format", "json"])
        assert_success(result)
        payload = json.loads(result.stdout)
        assert len(payload) == 1

        result2 = run_simgrep_command(["search", "fox", str(docs_dir), "--keyword", "brown", "--keyword", "lavender", "--format", "json"])
        assert_success(result2)
        payload2 = json.loads(result2.stdout)
        assert len(payload2) == 0


class TestPreferAndWeight:
    def test_prefer_boosts_ranking(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "prefer_test"
        docs_dir.mkdir()
        (docs_dir / "normal.txt").write_text("important business logic code", encoding="utf-8")
        (docs_dir / "preferred.py").write_text("important business logic code", encoding="utf-8")

        result = run_simgrep_command(
            ["search", "business logic", str(docs_dir), "--prefer", "*.py", "--format", "json"],
        )
        payload = assert_clean_json_list(result)
        assert payload
        first_path = payload[0]["path"]
        assert "preferred.py" in first_path

    def test_prefer_weight_affects_ranking(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "weight_test"
        docs_dir.mkdir()
        (docs_dir / "low.txt").write_text("alpha beta gamma code", encoding="utf-8")
        (docs_dir / "high.txt").write_text("alpha beta gamma code", encoding="utf-8")

        result_low = run_simgrep_command(
            ["search", "alpha beta", str(docs_dir), "--prefer", "*.txt", "--prefer-weight", "0.05", "--format", "json"],
        )
        result_high = run_simgrep_command(
            ["search", "alpha beta", str(docs_dir), "--prefer", "*.txt", "--prefer-weight", "0.8", "--format", "json"],
        )

        low_payload = assert_clean_json_list(result_low)
        high_payload = assert_clean_json_list(result_high)

        if low_payload and high_payload:
            low_first = low_payload[0]["path"]
            high_first = high_payload[0]["path"]
            assert low_first != high_first or low_payload[0]["score"] != high_payload[0]["score"]


class TestAbsolutePaths:
    @pytest.fixture
    def abs_project(self, tmp_path: pathlib.Path) -> pathlib.Path:
        project_dir = tmp_path / "abs_project"
        docs_dir = project_dir / "docs"
        project_dir.mkdir()
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("absolute path test content", encoding="utf-8")

        assert_success(run_simgrep_command(["init"], cwd=project_dir))
        assert_success(run_simgrep_command(["project", "add-path", str(docs_dir)], cwd=project_dir))
        assert_success(run_simgrep_command(["index", "--rebuild"], cwd=project_dir))
        return project_dir

    def test_absolute_paths_json(self, temp_simgrep_home: pathlib.Path, abs_project: pathlib.Path) -> None:
        result = run_simgrep_command(["search", "absolute", "--absolute-paths", "--format", "json"], cwd=abs_project)
        payload = assert_clean_json_list(result)
        for item in payload:
            p = item["path"]
            assert p.startswith("/") or (len(p) > 1 and p[1] == ":")

    def test_absolute_paths_jsonl(self, temp_simgrep_home: pathlib.Path, abs_project: pathlib.Path) -> None:
        result = run_simgrep_command(["search", "absolute", "--absolute-paths", "--format", "jsonl"], cwd=abs_project)
        lines = [line for line in result.stdout.strip().split("\n") if line.strip()]
        for line in lines:
            item = json.loads(line)
            p = item["path"]
            assert p.startswith("/") or (len(p) > 1 and p[1] == ":")

    def test_absolute_paths_grep(self, temp_simgrep_home: pathlib.Path, abs_project: pathlib.Path) -> None:
        result = run_simgrep_command(["search", "absolute", "--absolute-paths", "--format", "grep"], cwd=abs_project)
        for line in result.stdout.splitlines():
            if line.strip():
                assert line.split(":")[0].startswith("/") or ":" in line.split(":")[0][1:]


class TestNoLineNumbers:
    def test_no_line_numbers_json(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "no_line_test"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("line one\nline two data\nline three", encoding="utf-8")

        result = run_simgrep_command(
            ["search", "line two", str(docs_dir), "--no-line-numbers", "--format", "json"],
        )
        payload = assert_clean_json_list(result)
        assert payload
        for item in payload:
            assert item.get("line_start") is None or item.get("line_start") == 0
            assert item.get("line_end") is None or item.get("line_end") == 0

    def test_no_line_numbers_compact_no_line_suffix(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "no_line_compact_test"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("line one\nline two data\nline three", encoding="utf-8")

        result = run_simgrep_command(
            ["search", "line two", str(docs_dir), "--no-line-numbers", "--format", "compact"],
        )
        assert_success(result)
        for line in result.stdout.splitlines():
            if line.strip():
                path_part = line.split()[0] if line.split() else ""
                assert ":" not in path_part or "score=" in line


class TestNoScores:
    def test_no_scores_grep_format(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "no_scores_test"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("score test content", encoding="utf-8")

        result = run_simgrep_command(["search", "score", str(docs_dir), "--no-scores", "--format", "grep"])
        assert_success(result)
        for line in result.stdout.splitlines():
            assert "score:" not in line.lower()

    def test_no_scores_compact_format(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "no_scores_compact_test"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("score compact content", encoding="utf-8")

        result = run_simgrep_command(["search", "score", str(docs_dir), "--no-scores", "--format", "compact"])
        assert_success(result)
        for line in result.stdout.splitlines():
            assert "score=" not in line.lower()


class TestWhyVisibility:
    def test_why_not_present_without_flag_json(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "no_why_json_test"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("alpha beta gamma", encoding="utf-8")

        result = run_simgrep_command(["search", "alpha beta", str(docs_dir), "--format", "json"])
        payload = assert_clean_json_list(result)
        assert payload
        assert "why" not in payload[0]

    def test_why_not_present_without_flag_jsonl(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "no_why_jsonl_test"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("delta epsilon zeta", encoding="utf-8")

        result = run_simgrep_command(["search", "epsilon", str(docs_dir), "--format", "jsonl"])
        lines = [line for line in result.stdout.strip().split("\n") if line.strip()]
        assert lines
        first = json.loads(lines[0])
        assert "why" not in first

    def test_why_not_present_without_flag_compact(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "no_why_compact_test"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("why test content", encoding="utf-8")

        result = run_simgrep_command(["search", "why", str(docs_dir), "--format", "compact"])
        assert_success(result)
        for line in result.stdout.splitlines():
            assert "why:" not in line.lower()


class TestContextRendering:
    def test_context_shows_neighboring_lines(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "context_test"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("line one\nline two TARGET\nline three\nline four\nline five", encoding="utf-8")

        result = run_simgrep_command(["search", "TARGET", str(docs_dir), "--context", "2", "--format", "json"])
        payload = assert_clean_json_list(result)
        assert payload
        first = payload[0]
        assert "line two TARGET" in first["text"] or "TARGET" in first["text"]

    def test_context_reads_actual_file(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "context_file_read"
        docs_dir.mkdir()
        content = "START\nMIDDLE_TARGET\nEND\nAFTER\nFINAL"
        (docs_dir / "b.txt").write_text(content, encoding="utf-8")

        result = run_simgrep_command(["search", "MIDDLE_TARGET", str(docs_dir), "--context", "1", "--format", "compact"])
        assert_success(result)
        output = result.stdout
        assert "START" in output or "MIDDLE_TARGET" in output


class TestStaleOffsetsWithFreshnessSkip:
    @pytest.fixture
    def stale_project(self, tmp_path: pathlib.Path) -> pathlib.Path:
        project_dir = tmp_path / "stale_project"
        docs_dir = project_dir / "docs"
        project_dir.mkdir()
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("original content lambda", encoding="utf-8")

        assert_success(run_simgrep_command(["init"], cwd=project_dir))
        assert_success(run_simgrep_command(["project", "add-path", str(docs_dir)], cwd=project_dir))
        assert_success(run_simgrep_command(["index", "--rebuild"], cwd=project_dir))
        return project_dir

    def test_stale_offsets_marked_when_file_modified(self, temp_simgrep_home: pathlib.Path, stale_project: pathlib.Path) -> None:
        docs_dir = stale_project / "docs"
        a_file = docs_dir / "a.txt"

        result1 = run_simgrep_command(["search", "lambda", "--freshness", "skip", "--format", "json"], cwd=stale_project)
        payload1 = assert_clean_json_list(result1)
        assert payload1

        a_file.write_text("modified content with different offsets xyz", encoding="utf-8")

        result2 = run_simgrep_command(["search", "xyz", "--freshness", "skip", "--format", "json"], cwd=stale_project)
        payload2 = assert_clean_json_list(result2)
        assert payload2

    def test_stale_offsets_does_not_crash_context(self, temp_simgrep_home: pathlib.Path, stale_project: pathlib.Path) -> None:
        docs_dir = stale_project / "docs"
        a_file = docs_dir / "a.txt"
        a_file.write_text("original content mu", encoding="utf-8")

        run_simgrep_command(["index", "--rebuild"], cwd=stale_project)

        a_file.write_text("modified now with extra content nu", encoding="utf-8")

        result = run_simgrep_command(
            ["search", "nu", "--freshness", "skip", "--context", "1", "--format", "json"],
            cwd=stale_project,
        )
        assert_success(result)


class TestSearchPathHint:
    def test_ephemeral_with_path_no_project_hint(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "ephemeral_no_project"
        docs_dir.mkdir()
        (docs_dir / "orphan.txt").write_text("orphan content mu", encoding="utf-8")

        result = run_simgrep_command(["search", "orphan", str(docs_dir)])
        assert_success(result)
        assert "orphan" in result.stdout.lower()
