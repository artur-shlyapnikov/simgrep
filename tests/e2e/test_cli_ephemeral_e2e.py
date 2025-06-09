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
        assert "Processing:" in result.stdout
        assert "100%" in result.stdout
        assert validation_fn(result)

    def test_ephemeral_search_single_file_paths_mode(self, tmp_path: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
        file_path = tmp_path / "single.txt"
        file_path.write_text("grapefruit and apples")

        result = run_simgrep_command(["search", "grapefruit", str(file_path), "--output", "paths"])
        assert result.exit_code == 0
        assert "single.txt" in result.stdout
        assert "Processing:" in result.stdout
        assert "100%" in result.stdout

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

    def test_ephemeral_search_json_output(self, tmp_path: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
        docs_dir = tmp_path / "ephemeral_docs_json"
        docs_dir.mkdir()
        file1 = docs_dir / "data.txt"
        file1.write_text("some interesting data for json output")

        result = run_simgrep_command(["search", "interesting json", str(docs_dir), "--output", "json"])
        assert result.exit_code == 0

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
            pytest.fail("--output json did not produce valid JSON")

    def test_ephemeral_search_no_matches(self, ephemeral_docs_dir: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
        result = run_simgrep_command(["search", "xyz", str(ephemeral_docs_dir), "--output", "count", "--min-score", "0.9"])
        assert result.exit_code == 0
        assert "0 matching chunks in 0 files." in result.stdout
