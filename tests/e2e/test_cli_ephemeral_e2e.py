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
