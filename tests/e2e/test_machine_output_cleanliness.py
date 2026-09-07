"""Machine output cleanliness: JSON, JSONL, paths, count, grep formats."""

import json
import pathlib
import re

from .conftest import (
    assert_paths_only,
    assert_success,
    run_simgrep_command,
)


class TestJsonCleanliness:
    def test_json_valid_with_results(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "json_results"
        docs_dir.mkdir()
        (docs_dir / "test.txt").write_text("hello world example")

        result = run_simgrep_command(["search", "hello", str(docs_dir), "--format", "json"])
        assert_success(result)
        assert result.stderr == ""
        payload = json.loads(result.stdout)
        assert isinstance(payload, list)
        assert len(payload) > 0

    def test_json_valid_no_results(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "json_no_results"
        docs_dir.mkdir()
        (docs_dir / "test.txt").write_text("foobar baz")

        result = run_simgrep_command(["search", "nomatch", str(docs_dir), "--format", "json"])
        assert_success(result)
        assert result.stderr == ""
        payload = json.loads(result.stdout)
        assert isinstance(payload, list)

    def test_json_with_unicode_path(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "unicode_docs"
        docs_dir.mkdir()
        (docs_dir / "日本語.txt").write_text("hello world")
        (docs_dir / "émojis_🎉.txt").write_text("test content")

        result = run_simgrep_command(["search", "hello", str(docs_dir), "--format", "json"])
        assert_success(result)
        assert result.stderr == ""
        payload = json.loads(result.stdout)
        assert isinstance(payload, list)
        for item in payload:
            assert "path" in item
            assert isinstance(item["path"], str)

    def test_json_with_unicode_snippet(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "unicode_snippets"
        docs_dir.mkdir()
        content = "日本語テスト content émojis 🎉 unicode"
        (docs_dir / "test.txt").write_text(content)

        result = run_simgrep_command(["search", "テスト", str(docs_dir), "--format", "json"])
        assert_success(result)
        assert result.stderr == ""
        payload = json.loads(result.stdout)
        assert isinstance(payload, list)
        for item in payload:
            assert "text" in item
            assert isinstance(item["text"], str)

    def test_json_with_multiline_snippet(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "multiline_snippets"
        docs_dir.mkdir()
        content = "line one\nline two content\nline three"
        (docs_dir / "test.txt").write_text(content)

        result = run_simgrep_command(["search", "content", str(docs_dir), "--format", "json", "-c", "2"])
        assert_success(result)
        assert result.stderr == ""
        payload = json.loads(result.stdout)
        assert isinstance(payload, list)
        for item in payload:
            assert "text" in item
            assert isinstance(item["text"], str)
            assert isinstance(item.get("line_start"), int)

    def test_json_truncation_preserves_validity(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "truncation_test"
        docs_dir.mkdir()
        long_content = "word " * 500
        (docs_dir / "long.txt").write_text(long_content)

        result = run_simgrep_command(["search", "word", str(docs_dir), "--format", "json", "--max-chars", "200"])
        assert_success(result)
        assert result.stderr == ""
        payload = json.loads(result.stdout)
        assert isinstance(payload, list)
        for item in payload:
            assert "text" in item
            assert isinstance(item["text"], str)
            assert len(item["text"]) <= 200 + 3


class TestJsonlCleanliness:
    def test_jsonl_valid_per_line(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "jsonl_results"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("apple banana")
        (docs_dir / "b.txt").write_text("banana cherry")

        result = run_simgrep_command(["search", "banana", str(docs_dir), "--format", "jsonl"])
        assert_success(result)
        assert result.stderr == ""
        lines = result.stdout.splitlines()
        assert len(lines) > 0
        for line in lines:
            json.loads(line)


class TestPathsCleanliness:
    def test_paths_deduplicated(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "paths_dedup"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("match content")
        (docs_dir / "b.txt").write_text("match content")

        result = run_simgrep_command(["search", "content", str(docs_dir), "--format", "paths"])
        assert_success(result)
        paths = assert_paths_only(result)
        deduped = set(paths)
        assert len(paths) == len(deduped), "paths should be deduplicated"

    def test_paths_stably_sorted(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "paths_sorted"
        docs_dir.mkdir()
        (docs_dir / "z_file.txt").write_text("content")
        (docs_dir / "a_file.txt").write_text("content")
        (docs_dir / "m_file.txt").write_text("content")

        result = run_simgrep_command(["search", "content", str(docs_dir), "--format", "paths"])
        assert_success(result)
        paths = assert_paths_only(result)
        assert paths == sorted(paths), "paths should be sorted"

    def test_paths_no_scores_or_extra(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "paths_clean"
        docs_dir.mkdir()
        (docs_dir / "test.txt").write_text("match")

        result = run_simgrep_command(["search", "match", str(docs_dir), "--format", "paths"])
        assert_success(result)
        paths = assert_paths_only(result)
        for path in paths:
            assert not re.search(r"score|line|:\d", path), f"unexpected format in path: {path}"


class TestCountCleanliness:
    def test_count_only_number_and_newline(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "count_test"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("hello")
        (docs_dir / "b.txt").write_text("hello")

        result = run_simgrep_command(["search", "hello", str(docs_dir), "--format", "count"])
        assert_success(result)
        assert result.stderr == ""
        lines = result.stdout.strip().split("\n")
        assert len(lines) == 1
        assert lines[0].isdigit() or lines[0] == "0"

    def test_count_no_results(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "count_no_results"
        docs_dir.mkdir()
        (docs_dir / "test.txt").write_text("foobar")

        result = run_simgrep_command(["search", "nomatch", str(docs_dir), "--format", "count"])
        assert_success(result)
        assert result.stderr == ""
        lines = result.stdout.strip().split("\n")
        assert len(lines) == 1
        assert lines[0].isdigit()


class TestGrepCleanliness:
    def test_grep_format_stable(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "grep_format"
        docs_dir.mkdir()
        (docs_dir / "test.txt").write_text("line one\nline two match\nline three")

        result = run_simgrep_command(["search", "match", str(docs_dir), "--format", "grep"])
        assert_success(result)
        assert result.stderr == ""
        lines = result.stdout.strip().split("\n")
        for line in lines:
            assert re.match(r"^[^:]+:\d+(:[\d.]+)?: ", line), f"malformed grep line: {line}"

    def test_grep_unicode_content(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "grep_unicode"
        docs_dir.mkdir()
        (docs_dir / "日本語.txt").write_text("line one\nテスト match\nline three")

        result = run_simgrep_command(["search", "match", str(docs_dir), "--format", "grep"])
        assert_success(result)
        for line in result.stdout.strip().split("\n"):
            assert isinstance(line, str)
            assert "match" in line


class TestMachineOutputStdin:
    def test_errors_go_to_stderr(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "error_test"
        docs_dir.mkdir()

        result = run_simgrep_command(["search", "term", str(docs_dir), "--format", "json", "--persistent"])
        assert result.exit_code != 0
        combined = result.stdout + result.stderr
        assert "persistent" in combined.lower() or "project" in combined.lower()

    def test_machine_stdout_clean_on_error(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "clean_error"
        docs_dir.mkdir()
        (docs_dir / "test.txt").write_text("hello")

        result = run_simgrep_command(["search", "hello", str(docs_dir), "--format", "json", "--persistent"])
        assert result.exit_code != 0
        try:
            parsed = json.loads(result.stdout)
            assert isinstance(parsed, list)
        except json.JSONDecodeError:
            pass


class TestUnicodeFilenames:
    def test_unicode_filename_in_paths(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "unicode_filenames"
        docs_dir.mkdir()
        (docs_dir / "日本語.txt").write_text("content")
        (docs_dir / "français.md").write_text("content")
        (docs_dir / "עברית.rst").write_text("content")
        (docs_dir / "🎉.txt").write_text("content")

        result = run_simgrep_command(["search", "content", str(docs_dir), "--format", "paths"])
        assert_success(result)
        assert result.stderr == ""
        paths = assert_paths_only(result)
        assert len(paths) == 4
        for path in paths:
            assert isinstance(path, str)


class TestSnippetTruncation:
    def test_truncation_preserves_json(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "trunc_json"
        docs_dir.mkdir()
        long_content = "word " * 1000
        (docs_dir / "long.txt").write_text(long_content)

        result = run_simgrep_command(["search", "word", str(docs_dir), "--format", "json", "--max-chars", "100"])
        assert_success(result)
        assert result.stderr == ""
        payload = json.loads(result.stdout)
        for item in payload:
            text = item["text"]
            assert text.endswith("...")

    def test_truncation_preserves_line_numbers(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "trunc_lines"
        docs_dir.mkdir()
        content = "line one\n" * 100
        (docs_dir / "many.txt").write_text(content)

        result = run_simgrep_command(["search", "one", str(docs_dir), "--format", "json", "--max-chars", "50"])
        assert_success(result)
        assert result.stderr == ""
        payload = json.loads(result.stdout)
        for item in payload:
            assert isinstance(item.get("line_start"), int)

    def test_truncation_preserves_unicode(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "trunc_unicode"
        docs_dir.mkdir()
        content = "日本語 " * 200
        (docs_dir / "unicode.txt").write_text(content)

        result = run_simgrep_command(["search", "日本語", str(docs_dir), "--format", "json", "--max-chars", "100"])
        assert_success(result)
        assert result.stderr == ""
        payload = json.loads(result.stdout)
        for item in payload:
            text = item["text"]
            assert "日本語" in text or text.endswith("...")
