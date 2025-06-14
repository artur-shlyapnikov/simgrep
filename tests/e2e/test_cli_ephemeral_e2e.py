import json
import os
import pathlib
from typing import Callable

import pytest
from typer.testing import Result

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
    @pytest.fixture(autouse=True)
    def _init_global_for_e2e(self, temp_simgrep_home: pathlib.Path) -> None:
        """
        Most E2E tests require a global config to exist, even if they don't
        use a persistent project, due to model caching etc.
        This fixture ensures the global config is present for all tests in the class.
        """
        run_simgrep_command(["init", "--global"])

    @pytest.fixture
    def ephemeral_docs_dir(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """Creates a directory with sample files for ephemeral search tests."""
        docs_dir = tmp_path / "ephemeral_docs"
        docs_dir.mkdir()
        (docs_dir / "one.txt").write_text("apples bananas")
        (docs_dir / "two.txt").write_text("bananas oranges")
        return docs_dir

    @pytest.mark.parametrize(
        "output_mode, extra_args, validation_fn",
        [
            pytest.param(
                "show",
                [],
                lambda r: "File:" in r.stdout and "one.txt" in r.stdout and "two.txt" in r.stdout,
                id="show_mode",
            ),
            pytest.param(
                "paths",
                [],
                lambda r: "one.txt" in r.stdout and "two.txt" in r.stdout,
                id="paths_mode",
            ),
            pytest.param(
                "count",
                ["--min-score", "0.5"],
                lambda r: "2 matching chunks in 2 files" in r.stdout,
                id="count_mode",
            ),
        ],
    )
    def test_ephemeral_search_output_modes(
        self,
        ephemeral_docs_dir: pathlib.Path,
        temp_simgrep_home: pathlib.Path,
        output_mode: str,
        extra_args: list[str],
        validation_fn: Callable[[Result], bool],
    ) -> None:
        """Tests various output modes for ephemeral search."""
        args = ["search", "bananas", str(ephemeral_docs_dir), "--output", output_mode] + extra_args
        result = run_simgrep_command(args)

        assert result.exit_code == 0
        if output_mode == "show":
            assert "Processing:" in result.stdout
            assert "100%" in result.stdout
        assert validation_fn(result)

    def test_ephemeral_search_single_file_paths_mode(self, tmp_path: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
        file_path = tmp_path / "single.txt"
        file_path.write_text("grapefruit and apples")

        result = run_simgrep_command(["search", "grapefruit", str(file_path), "--output", "paths"])
        assert result.exit_code == 0
        assert "single.txt" in result.stdout

    def test_ephemeral_search_single_file_relative_path(self, tmp_path: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
        """
        Tests that --output paths --relative-paths on a single file target
        produces a path relative to the file's parent directory.
        """
        file_path = tmp_path / "single_relative.txt"
        file_path.write_text("grapefruit and apples")

        result = run_simgrep_command(
            [
                "search",
                "grapefruit",
                str(file_path),
                "--output",
                "paths",
                "--relative-paths",
            ]
        )
        assert result.exit_code == 0
        # The output should be just the filename, with a newline.
        # With the fixes to suppress logging, this should be the only output
        assert result.stdout.strip() == "single_relative.txt"

    def test_ephemeral_search_relative_paths(self, tmp_path: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
        docs_dir = tmp_path / "docs_rel"
        docs_dir.mkdir()
        (docs_dir / "root.txt").write_text("apples here")
        subdir = docs_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("more apples")

        result = run_simgrep_command(
            [
                "search",
                "apples",
                str(docs_dir),
                "--output",
                "paths",
                "--relative-paths",
            ],
        )
        assert result.exit_code == 0
        assert "root.txt" in result.stdout
        assert os.path.join("subdir", "nested.txt") in result.stdout

    def test_ephemeral_search_json_output_is_clean_and_valid(self, tmp_path: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
        """
        Tests that --output json is clean and machine-readable, with no
        logging or other text on stdout or stderr, and is valid JSON.
        """
        docs_dir = tmp_path / "ephemeral_docs_json"
        docs_dir.mkdir()
        file1 = docs_dir / "data.txt"
        file1.write_text("some interesting data for json output")

        result = run_simgrep_command(["search", "interesting json", str(docs_dir), "--output", "json"])
        assert result.exit_code == 0
        assert result.stderr == ""

        try:
            json_output = json.loads(result.stdout)
            assert isinstance(json_output, list)
            assert len(json_output) >= 1
            item = json_output[0]
            assert "file_path" in item
            assert "data.txt" in item["file_path"]
            assert "score" in item
            assert "chunk_text" in item
            assert "interesting data" in item["chunk_text"]
        except json.JSONDecodeError:
            pytest.fail(f"--output json did not produce valid JSON. Output:\n{result.stdout}")

    def test_ephemeral_search_paths_output_is_clean(self, ephemeral_docs_dir: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
        """
        Tests that --output paths is clean and machine-readable, with no
        logging or other text on stdout or stderr.
        """
        args = ["search", "bananas", str(ephemeral_docs_dir), "--output", "paths"]
        result = run_simgrep_command(args)

        assert result.exit_code == 0
        # With machine-readable output, stderr should be clean
        assert result.stderr == ""

        lines = result.stdout.strip().split("\n")
        assert len(lines) == 2  # one.txt, two.txt
        # Sort for deterministic check
        lines.sort()
        assert "one.txt" in lines[0]
        assert "two.txt" in lines[1]
        for line in lines:
            assert line.endswith(".txt")
            assert not line.startswith(" ")
            assert pathlib.Path(line).is_absolute()

    def test_ephemeral_search_no_matches(self, ephemeral_docs_dir: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
        result = run_simgrep_command(["search", "xyz", str(ephemeral_docs_dir), "--output", "count", "--min-score", "0.9"])
        assert result.exit_code == 0
        assert "0 matching chunks in 0 files." in result.stdout

    def test_ephemeral_search_on_nonexistent_path_fails_early(self, tmp_path: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
        """Tests that ephemeral search fails with a clear error for a non-existent path."""
        non_existent_path = tmp_path / "this_path_does_not_exist"
        args = ["search", "query", str(non_existent_path)]
        result = run_simgrep_command(args)
        assert result.exit_code != 0
        # Typer provides this error message for arguments with exists=True
        assert "does not exist" in result.stderr
        assert "this_path_does_not_exist" in result.stderr

    def test_ephemeral_search_multiple_patterns(self, tmp_path: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
        """
        Tests ephemeral search with multiple --pattern arguments to include different file types.
        """
        docs_dir = tmp_path / "multi_pattern_docs"
        docs_dir.mkdir()
        (docs_dir / "doc.py").write_text("searchable_term_in_python = True")
        (docs_dir / "doc.md").write_text("# searchable_term_in_markdown")
        (docs_dir / "doc.txt").write_text("this file should be ignored")

        args = [
            "search",
            "searchable_term",
            str(docs_dir),
            "--pattern",
            "*.py",
            "--pattern",
            "*.md",
            "--output",
            "paths",
        ]
        result = run_simgrep_command(args)

        assert result.exit_code == 0
        assert "doc.py" in result.stdout
        assert "doc.md" in result.stdout
        assert "doc.txt" not in result.stdout