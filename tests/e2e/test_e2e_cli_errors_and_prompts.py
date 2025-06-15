import os
import pathlib
from typing import Generator
from unittest.mock import patch

import pytest

from simgrep.core.errors import IndexerError, SimgrepConfigError, SimgrepError

from .conftest import populated_persistent_index, run_simgrep_command


@pytest.fixture
def temp_project_dir(tmp_path: pathlib.Path) -> Generator[pathlib.Path, None, None]:
    """Creates a temporary, isolated project directory."""
    proj_dir = tmp_path / "my-project"
    proj_dir.mkdir()
    original_cwd = os.getcwd()
    os.chdir(proj_dir)
    yield proj_dir
    os.chdir(original_cwd)


class TestCliErrorsAndPromptsE2E:
    def test_search_on_unindexed_project_fails_gracefully(
        self, temp_simgrep_home: pathlib.Path
    ) -> None:
        """Verify helpful message when searching a project that exists but has not been indexed."""
        # Arrange: init global, create project, but do not index
        run_simgrep_command(["init", "--global"], cwd=temp_simgrep_home)
        run_simgrep_command(
            ["project", "create", "my-empty-project"], cwd=temp_simgrep_home
        )

        # Act
        result = run_simgrep_command(
            ["search", "query", "--project", "my-empty-project"], cwd=temp_simgrep_home
        )

        # Assert
        assert result.exit_code == 1
        assert (
            "Persistent index for project 'my-empty-project' not found" in result.stdout
        )
        assert (
            "Please run 'simgrep index --project my-empty-project' first"
            in result.stdout
        )

    def test_index_rebuild_prompt_aborts_on_no(
        self, populated_persistent_index: None, sample_docs_dir_session: pathlib.Path
    ) -> None:
        """Ensure --rebuild prompt correctly aborts when user inputs 'n'."""
        # Arrange: Get baseline status from populated index
        status_before = run_simgrep_command(["status"], cwd=sample_docs_dir_session)
        assert status_before.exit_code == 0
        assert "files indexed" in status_before.stdout

        # Act: Run index --rebuild and provide 'n' as input
        result = run_simgrep_command(
            ["index", "--rebuild"], cwd=sample_docs_dir_session, input_str="n\n"
        )

        # Assert
        assert "Aborted" in result.stderr
        assert result.exit_code != 0  # typer.Abort raises SystemExit(1)

        # Verify no data was wiped
        status_after = run_simgrep_command(["status"], cwd=sample_docs_dir_session)
        assert status_after.exit_code == 0
        assert status_after.stdout == status_before.stdout

    def test_init_in_dir_where_project_name_already_exists_globally(
        self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path
    ) -> None:
        """Verify `simgrep init` fails if a project with the same name already exists."""
        # Arrange: Create a global project
        run_simgrep_command(["init", "--global"], cwd=temp_simgrep_home)
        run_simgrep_command(["project", "create", "my-project"], cwd=temp_simgrep_home)

        # Create a directory with the same name
        project_dir = tmp_path / "my-project"
        project_dir.mkdir()

        # Act: Try to initialize a project in that directory
        result = run_simgrep_command(["init"], cwd=project_dir)

        # Assert
        assert result.exit_code == 1
        assert "A project named 'my-project' already exists globally" in result.stdout

    def test_ephemeral_search_with_unreadable_file_is_skipped(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Verify that ephemeral search handles unreadable files gracefully."""
        # Arrange
        search_dir = tmp_path / "search_dir"
        search_dir.mkdir()
        readable_file = search_dir / "readable.txt"
        readable_file.write_text("This is a readable file with content.")
        unreadable_file = search_dir / "unreadable.txt"
        unreadable_file.write_text("This is unreadable.")
        os.chmod(unreadable_file, 0o000)

        # Act
        result = run_simgrep_command(
            ["search", "content", str(search_dir), "--pattern", "*.txt"]
        )

        # Assert
        assert result.exit_code == 0
        assert "Chunk: this is a readable file with content." in result.stdout
        assert "This is unreadable." not in result.stdout
        # The warning is now logged by unstructured and not printed to console

        # Restore permissions to allow cleanup
        os.chmod(unreadable_file, 0o644)

    def test_index_fails_gracefully_on_indexer_error(
        self, temp_simgrep_home: pathlib.Path, populated_persistent_index: None
    ) -> None:
        """Cover the IndexerError handling block in the index command."""
        from simgrep.indexer import IndexerError

        with patch(
            "simgrep.main.Indexer.run_index",
            side_effect=IndexerError("mock indexer error"),
        ):
            result = run_simgrep_command(["index"])
            assert result.exit_code == 1
            assert "Indexing Error:" in result.stdout
            assert "mock indexer error" in result.stdout

    def test_search_persistent_handles_simgrep_error(
        self, temp_simgrep_home: pathlib.Path, populated_persistent_index: None
    ) -> None:
        """Cover the generic SimgrepError handling in the search command for persistent search."""
        from simgrep.core.errors import SimgrepError

        with patch(
            "simgrep.main.SearchService.search",
            side_effect=SimgrepError("mock search error"),
        ):
            result = run_simgrep_command(["search", "query"])
            assert result.exit_code == 1
            assert "Error during persistent search: mock search error" in result.stdout

    def test_project_add_path_to_nonexistent_project(
        self, temp_simgrep_home: pathlib.Path
    ) -> None:
        """Cover the error path for project add-path when the project doesn't exist."""
        run_simgrep_command(["init", "--global"])
        result = run_simgrep_command(
            ["project", "add-path", ".", "--project", "non-existent-project"]
        )
        assert result.exit_code == 1
        assert (
            "Error adding path to project: Project 'non-existent-project' not found."
            in result.stdout
        )

    def test_index_with_workers_option(
        self, populated_persistent_index: None, sample_docs_dir_session: pathlib.Path
    ) -> None:
        """Ensure the --workers option is correctly passed and used."""
        # This test just ensures the command doesn't crash.
        # It's hard to verify concurrency from an E2E test.
        result = run_simgrep_command(
            ["index", "--workers", "1"], cwd=sample_docs_dir_session
        )
        assert result.exit_code == 0
        assert "Successfully indexed" in result.stdout

    def test_index_rebuild_with_yes_flag_skips_prompt(
        self, populated_persistent_index: None, sample_docs_dir_session: pathlib.Path
    ) -> None:
        """Verify that the --yes flag bypasses the confirmation prompt for --rebuild."""
        result = run_simgrep_command(
            ["index", "--rebuild", "--yes"], cwd=sample_docs_dir_session
        )
        assert result.exit_code == 0
        assert "Are you sure" not in result.stdout
        assert "Aborted" not in result.stderr
        assert "Successfully indexed" in result.stdout
