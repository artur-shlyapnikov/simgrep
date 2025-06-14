import pathlib

import pytest

from .test_cli_persistent_e2e import run_simgrep_command
from .test_cli_persistent_e2e import temp_simgrep_home as _temp_simgrep_home

# Re-export fixture for pytest to discover and use
temp_simgrep_home = _temp_simgrep_home

pytest.importorskip("sentence_transformers")
pytest.importorskip("usearch.index")


@pytest.fixture
def docs_with_gitignore(tmp_path: pathlib.Path) -> pathlib.Path:
    """
    Creates a directory structure with a .gitignore file for testing ignore logic.
    """
    project_dir = tmp_path / "gitignore_project"
    project_dir.mkdir()

    # .gitignore file
    (project_dir / ".gitignore").write_text(
        """
# Ignore specific files
ignored_file.txt

# Ignore directories
ignored_dir/

# Ignore by pattern
*.log
    """
    )

    # Files that should be processed
    (project_dir / "regular_file.txt").write_text("This is a regular file with searchable content.")

    # Files that should be ignored
    (project_dir / "ignored_file.txt").write_text("This file should be ignored by name.")
    (project_dir / "system.log").write_text("This log file should be ignored by pattern.")

    # Directory that should be ignored
    ignored_dir = project_dir / "ignored_dir"
    ignored_dir.mkdir()
    (ignored_dir / "another_file.txt").write_text("This file is inside an ignored directory.")

    # Subdirectory with its own files
    sub_dir = project_dir / "subdir"
    sub_dir.mkdir()
    (sub_dir / "sub_file.txt").write_text("A file in a subdirectory, also searchable.")
    (sub_dir / "another.log").write_text("A log file in a subdirectory, should be ignored.")

    return project_dir


class TestCliGitignoreE2E:
    def test_ephemeral_search_respects_gitignore(self, docs_with_gitignore: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
        """
        Tests that ephemeral search (`simgrep search <path>`) correctly ignores files
        specified in a .gitignore file.
        """
        args = ["search", "searchable", str(docs_with_gitignore), "--output", "paths"]
        result = run_simgrep_command(args)

        assert result.exit_code == 0
        output = result.stdout

        # Assert that non-ignored files are found
        assert "regular_file.txt" in output
        assert "sub_file.txt" in output

        # Assert that ignored files are NOT found
        assert "ignored_file.txt" not in output
        assert "system.log" not in output
        assert "another.log" not in output
        assert "another_file.txt" not in output  # from ignored_dir

    def test_ephemeral_search_on_explicitly_ignored_file_is_honored(self, docs_with_gitignore: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
        """
        Tests that if a user provides a direct path to an ignored file, it is still
        processed, as the explicit user intent overrides the .gitignore.
        """
        ignored_file_path = docs_with_gitignore / "ignored_file.txt"
        args = [
            "search",
            "ignored by name",
            str(ignored_file_path),
            "--output",
            "paths",
        ]
        result = run_simgrep_command(args)

        assert result.exit_code == 0
        # The explicitly provided file should be the only one in the output
        assert "ignored_file.txt" in result.stdout
        assert "regular_file.txt" not in result.stdout

    def test_persistent_index_respects_gitignore(self, docs_with_gitignore: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
        """
        Tests that persistent indexing (`simgrep index`) correctly ignores files
        specified in a .gitignore file.
        """
        # 1. Global init and create project
        run_simgrep_command(["init", "--global"])
        run_simgrep_command(["project", "create", "ignore-test"])
        run_simgrep_command(
            [
                "project",
                "add-path",
                str(docs_with_gitignore),
                "--project",
                "ignore-test",
            ]
        )

        # 2. Index the project
        index_result = run_simgrep_command(["index", "--project", "ignore-test", "--rebuild"], input_str="y\n")

        assert index_result.exit_code == 0
        # Should process only 'regular_file.txt' and 'subdir/sub_file.txt'
        assert "2 files processed" in index_result.stdout
        assert "0 errors encountered" in index_result.stdout

        # 3. Search the project
        search_result = run_simgrep_command(["search", "searchable", "--project", "ignore-test", "--output", "paths"])

        assert search_result.exit_code == 0
        output = search_result.stdout

        # Assert that non-ignored files are found
        assert "regular_file.txt" in output
        assert "sub_file.txt" in output

        # Assert that ignored files are NOT found
        assert "ignored_file.txt" not in output
        assert "system.log" not in output

    def test_nested_gitignore_is_respected(self, tmp_path: pathlib.Path, temp_simgrep_home: pathlib.Path) -> None:
        """
        Tests that ignore rules from nested .gitignore files are correctly applied.
        """
        project_dir = tmp_path / "nested_ignore_project"
        project_dir.mkdir()

        # Root .gitignore
        (project_dir / ".gitignore").write_text("*.log\n")
        (project_dir / "main.log").write_text("root log")
        (project_dir / "important.txt").write_text("root important content")

        # Subdirectory with its own .gitignore
        sub_dir = project_dir / "sub"
        sub_dir.mkdir()
        (sub_dir / ".gitignore").write_text("*.tmp\n")
        (sub_dir / "data.tmp").write_text("temp data")
        (sub_dir / "sub_important.txt").write_text("sub important content")
        (sub_dir / "sub.log").write_text("sub log file")  # Should be ignored by root .gitignore

        # Ephemeral search
        args = ["search", "important", str(project_dir), "--output", "paths"]
        result = run_simgrep_command(args)

        assert result.exit_code == 0
        output = result.stdout

        assert "important.txt" in output
        assert "sub_important.txt" in output
        assert "main.log" not in output
        assert "data.tmp" not in output
        assert "sub.log" not in output
