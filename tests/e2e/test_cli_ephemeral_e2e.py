import pathlib

import pytest

from .test_cli_persistent_e2e import (
    run_simgrep_command,
)
from .test_cli_persistent_e2e import (
    temp_simgrep_home as _temp_simgrep_home,
)

temp_simgrep_home = _temp_simgrep_home  # re-export for pytest

pytest.importorskip("sentence_transformers")
pytest.importorskip("usearch.index")


class TestCliEphemeralE2E:
    def test_ephemeral_search_show_and_paths(self, tmp_path: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
        docs_dir = tmp_path / "ephemeral_docs"
        docs_dir.mkdir()
        file1 = docs_dir / "one.txt"
        file1.write_text("apples bananas")
        file2 = docs_dir / "two.txt"
        file2.write_text("bananas oranges")

        env_vars = {"HOME": str(temp_simgrep_home)}

        show_result = run_simgrep_command(["search", "bananas", str(docs_dir)], env=env_vars)
        assert show_result.returncode == 0
        assert "File:" in show_result.stdout
        assert "Processing:" in show_result.stdout
        assert "100%" in show_result.stdout

        paths_result = run_simgrep_command(["search", "bananas", str(docs_dir), "--output", "paths"], env=env_vars)
        assert paths_result.returncode == 0
        assert ".txt" in paths_result.stdout
        assert "Processing:" in paths_result.stdout
        assert "100%" in paths_result.stdout

    def test_ephemeral_search_single_file_paths_mode(self, tmp_path: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
        file_path = tmp_path / "single.txt"
        file_path.write_text("grapefruit and apples")

        env_vars = {"HOME": str(temp_simgrep_home)}

        result = run_simgrep_command(["search", "grapefruit", str(file_path), "--output", "paths"], env=env_vars)
        assert result.returncode == 0
        assert "single.txt" in result.stdout
        assert "Processing:" in result.stdout
        assert "100%" in result.stdout

    def test_ephemeral_search_relative_paths(self, tmp_path: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
        docs_dir = tmp_path / "docs_rel"
        docs_dir.mkdir()
        subdir = docs_dir / "sub"
        subdir.mkdir()
        file1 = docs_dir / "one.txt"
        file1.write_text("apples bananas")
        file2 = subdir / "two.txt"
        file2.write_text("bananas oranges")

        env_vars = {"HOME": str(temp_simgrep_home)}

        result = run_simgrep_command([
            "search",
            "bananas",
            str(docs_dir),
            "--output",
            "paths",
            "--relative-paths",
        ], env=env_vars)

        assert result.returncode == 0
        # Should show relative path to subdir/two.txt
        assert "sub/two.txt" in result.stdout

    def test_ephemeral_search_multiple_patterns(self, tmp_path: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
        docs_dir = tmp_path / "mixed_docs"
        docs_dir.mkdir()
        (docs_dir / "one.txt").write_text("alpha beta")
        (docs_dir / "two.md").write_text("gamma beta")

        env_vars = {"HOME": str(temp_simgrep_home)}
        result = run_simgrep_command([
            "search",
            "beta",
            str(docs_dir),
            "--pattern",
            "*.txt",
            "--pattern",
            "*.md",
            "--output",
            "paths",
        ], env=env_vars)

        assert result.returncode == 0
        assert "one.txt" in result.stdout
        assert "two.md" in result.stdout

    def test_ephemeral_search_top_limits_results(self, tmp_path: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
        docs_dir = tmp_path / "top_docs"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("apple orange banana")
        (docs_dir / "b.txt").write_text("banana kiwi lime")

        env_vars = {"HOME": str(temp_simgrep_home)}
        result = run_simgrep_command([
            "search",
            "banana",
            str(docs_dir),
            "--top",
            "1",
        ], env=env_vars)

        assert result.returncode == 0
        assert result.stdout.count("File:") == 1

    def test_ephemeral_search_no_results(self, tmp_path: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
        docs_dir = tmp_path / "empty_search_docs"
        docs_dir.mkdir()
        (docs_dir / "one.txt").write_text("alpha beta")

        env_vars = {"HOME": str(temp_simgrep_home)}
        result = run_simgrep_command([
            "search",
            "nomatchword",
            str(docs_dir),
        ], env=env_vars)

        assert result.returncode == 0
        assert "No relevant chunks found" in result.stdout

    def test_ephemeral_search_empty_directory(self, tmp_path: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
        empty_dir = tmp_path / "no_files"
        empty_dir.mkdir()

        env_vars = {"HOME": str(temp_simgrep_home)}
        result = run_simgrep_command(["search", "nothing", str(empty_dir)], env=env_vars)

        assert result.returncode == 0
        assert "No files found" in result.stdout or "No text chunks" in result.stdout
