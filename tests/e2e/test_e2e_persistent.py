import os
import pathlib
import sys
from typing import Callable, List
from unittest.mock import patch

import pytest
from click.testing import Result

from simgrep.config import SimgrepConfigError

from .conftest import _validate_json_output, run_simgrep_command

pytest.importorskip("sentence_transformers")
pytest.importorskip("usearch.index")


@pytest.mark.slow
class TestProjectWorkflowE2E:
    """
    Tests the core user workflows for creating, managing, and using persistent projects.
    """

    def test_global_init_and_project_create_list_add(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        """Tests creating a project, adding a path, and listing it."""
        # 1. Global init is required
        init_result = run_simgrep_command(["init", "--global"])
        assert init_result.exit_code == 0
        assert "Global simgrep configuration initialized" in init_result.stdout

        # 2. Create a new project
        create_result = run_simgrep_command(["project", "create", "my-new-project"])
        assert create_result.exit_code == 0
        assert "Project 'my-new-project' created." in create_result.stdout

        # 3. List projects and verify it's there
        list_result = run_simgrep_command(["project", "list"])
        assert list_result.exit_code == 0
        assert "default" in list_result.stdout
        assert "my-new-project" in list_result.stdout

        # 4. Add a path to the project
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        add_path_result = run_simgrep_command(["project", "add-path", str(docs_dir), "--project", "my-new-project"])
        assert add_path_result.exit_code == 0

    def test_project_create_fails_if_exists(self, temp_simgrep_home: pathlib.Path) -> None:
        """Tests that creating a project that already exists fails."""
        run_simgrep_command(["init", "--global"])
        run_simgrep_command(["project", "create", "my-project"])
        result = run_simgrep_command(["project", "create", "my-project"])
        assert result.exit_code == 1
        assert "Project 'my-project' already exists" in result.stdout

    def test_local_project_init_index_search_workflow(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        """
        Tests the core local project workflow: `init`, `index`, `search`
        all from within a project directory.
        """
        project_dir = tmp_path / "my-local-project"
        project_dir.mkdir()
        (project_dir / "file.txt").write_text("The quick brown fox jumps over the lazy dog.")

        # 1. Global init is a prerequisite
        run_simgrep_command(["init", "--global"])

        # 2. Local init inside the project directory
        init_result = run_simgrep_command(["init"], cwd=project_dir)
        assert init_result.exit_code == 0
        assert "Initialized simgrep project 'my-local-project'" in init_result.stdout
        assert (project_dir / ".simgrep").exists()

        # 3. Index the project (should be auto-detected)
        index_result = run_simgrep_command(["index", "--rebuild", "--yes"], cwd=project_dir)
        assert index_result.exit_code == 0
        assert "Successfully indexed project 'my-local-project'" in index_result.stdout

        # 4. Search the project (should be auto-detected)
        search_result = run_simgrep_command(["search", "lazy animal"], cwd=project_dir)
        assert search_result.exit_code == 0
        assert "lazy dog" in search_result.stdout
        assert "file.txt" in search_result.stdout

        # 5. Check status (should be auto-detected)
        status_result = run_simgrep_command(["status"], cwd=project_dir)
        assert status_result.exit_code == 0
        assert "Project 'my-local-project': 1 files indexed" in status_result.stdout

    def test_init_fails_if_already_initialized(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        """Tests that `simgrep init` in an already initialized directory fails."""
        project_dir = tmp_path / "already-init-proj"
        project_dir.mkdir()
        run_simgrep_command(["init", "--global"])
        run_simgrep_command(["init"], cwd=project_dir)

        # Run init again
        result = run_simgrep_command(["init"], cwd=project_dir)
        assert result.exit_code == 1
        assert "seems to be already initialized" in result.stdout

    def test_status_fails_on_non_existent_project(self, temp_simgrep_home: pathlib.Path) -> None:
        """Tests that `simgrep status` fails if the project does not exist."""
        run_simgrep_command(["init", "--global"])
        result = run_simgrep_command(["status", "--project", "non-existent-project"])
        assert result.exit_code == 1
        assert "Project 'non-existent-project' not found" in result.stdout

    def test_search_from_project_subdirectory_autodetects_project(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        """
        Tests that running commands from a subdirectory of an initialized project
        correctly autodetects and uses the parent project's context.
        """
        project_root = tmp_path / "my-codebase"
        project_subdir = project_root / "src" / "utils"
        project_subdir.mkdir(parents=True)
        (project_root / "README.md").write_text("Project about elegant systems.")

        run_simgrep_command(["init", "--global"])
        run_simgrep_command(["init"], cwd=project_root)
        run_simgrep_command(["project", "add-path", str(project_root)], cwd=project_root)
        index_result = run_simgrep_command(["index", "--rebuild", "--pattern", "*.md", "--yes"], cwd=project_root)
        assert index_result.exit_code == 0, index_result.stdout

        search_result = run_simgrep_command(["search", "elegant systems"], cwd=project_subdir)
        assert search_result.exit_code == 0
        assert "Detected project 'my-codebase'" in search_result.stdout
        assert "README.md" in search_result.stdout
        assert "elegant systems" in search_result.stdout

    def test_init_in_directory_with_spaces(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        project_dir = tmp_path / "My Awesome Project"
        project_dir.mkdir()
        (project_dir / "file.txt").write_text("Some content.")
        expected_project_name = "my-awesome-project"

        run_simgrep_command(["init", "--global"])
        init_result = run_simgrep_command(["init"], cwd=project_dir)
        assert init_result.exit_code == 0
        assert f"Initialized simgrep project '{expected_project_name}'" in init_result.stdout

        index_result = run_simgrep_command(["index", "--rebuild", "--yes"], cwd=project_dir)
        assert index_result.exit_code == 0, index_result.stdout
        assert f"Successfully indexed project '{expected_project_name}'" in index_result.stdout


@pytest.mark.slow
class TestPersistentSearchE2E:
    @pytest.mark.parametrize(
        "output_mode, extra_args, validation_fn",
        [
            pytest.param(
                "show",
                [],
                lambda r: "File:" in r.stdout and "doc1.txt" in r.stdout,
                id="show_mode",
            ),
            pytest.param("paths", [], lambda r: "doc1.txt" in r.stdout, id="paths_mode"),
            pytest.param("json", [], _validate_json_output, id="json_mode"),
            pytest.param(
                "count",
                ["--min-score", "0.4"],
                lambda r: "chunks in" in r.stdout,
                id="count_mode",
            ),
        ],
    )
    def test_search_output_modes(
        self,
        populated_persistent_index: None,
        output_mode: str,
        extra_args: List[str],
        validation_fn: Callable[[Result], bool],
    ) -> None:
        args = ["search", "apples", "--output", output_mode] + extra_args
        result = run_simgrep_command(args)
        assert result.exit_code == 0
        assert validation_fn(result)

    def test_search_subdirectories_and_no_matches(self, populated_persistent_index: None) -> None:
        search_banana_result = run_simgrep_command(["search", "bananas"])
        assert search_banana_result.exit_code == 0
        assert "doc1.txt" in search_banana_result.stdout
        assert os.path.join("subdir", "doc_sub.txt") in search_banana_result.stdout

        # Use a high min-score to ensure no matches are returned for this nonsensical query
        search_no_match_result = run_simgrep_command(["search", "nonexistentqueryxyz", "--min-score", "0.95"])
        assert search_no_match_result.exit_code == 0
        assert "No relevant chunks found" in search_no_match_result.stdout

    @pytest.mark.parametrize("top_k, expected_results", [(1, 1), (2, 2), (5, 3)])
    def test_search_top_option_limits_results(self, populated_persistent_index: None, top_k: int, expected_results: int) -> None:
        result = run_simgrep_command(["search", "apples", "--top", str(top_k)])
        assert result.exit_code == 0
        assert result.stdout.count("Score:") == expected_results

    def test_persistent_search_relative_paths(self, populated_persistent_index: None, sample_docs_dir_session: pathlib.Path) -> None:
        search_result = run_simgrep_command(
            ["search", "bananas", "--output", "paths", "--relative-paths"],
            cwd=sample_docs_dir_session,
        )
        assert search_result.exit_code == 0
        assert "doc1.txt" in search_result.stdout
        assert os.path.join("subdir", "doc_sub.txt") in search_result.stdout
        assert str(sample_docs_dir_session) not in search_result.stdout

    def test_search_with_filters(self, populated_persistent_index: None) -> None:
        args = [
            "search",
            "a document about bananas",
            "--file-filter",
            "*.txt",
            "--keyword",
            "kiwi",
            "--min-score",
            "0.5",
            "--output",
            "paths",
        ]
        result = run_simgrep_command(args)
        assert result.exit_code == 0
        assert "doc1.txt" in result.stdout
        assert "doc2.txt" not in result.stdout
        assert "doc_sub.txt" not in result.stdout


@pytest.mark.slow
class TestIncrementalIndexingE2E:
    def test_incremental_index_adds_new_file(
        self,
        populated_persistent_index_func_scope: None,
        sample_docs_dir_func_scope: pathlib.Path,
    ) -> None:
        new_file = sample_docs_dir_func_scope / "new_doc.txt"
        new_file.write_text("A new document about newly added content.")

        index_result = run_simgrep_command(["index"])
        assert index_result.exit_code == 0
        assert index_result.stdout.count("Skipped (unchanged)") == 3
        assert "Summary: 4 files processed" in index_result.stdout

        search_after_result = run_simgrep_command(["search", "newly added content"])
        assert search_after_result.exit_code == 0
        assert "new_doc.txt" in search_after_result.stdout

    def test_incremental_index_prunes_deleted_file(
        self,
        populated_persistent_index_func_scope: None,
        sample_docs_dir_func_scope: pathlib.Path,
    ) -> None:
        file_to_delete = sample_docs_dir_func_scope / "doc1.txt"
        content_to_disappear = "unique_fruit_kiwi"
        file_to_delete.unlink()
        assert not file_to_delete.exists()

        index_result = run_simgrep_command(["index"])
        assert index_result.exit_code == 0
        assert "Pruning 1 deleted file from index..." in index_result.stdout

        search_after_result = run_simgrep_command(["search", content_to_disappear, "--min-score", "0.5"])
        assert search_after_result.exit_code == 0
        assert "No relevant chunks found" in search_after_result.stdout

    def test_project_survives_directory_rename_and_reindex(
        self,
        populated_persistent_index_func_scope: None,
        sample_docs_dir_func_scope: pathlib.Path,
    ) -> None:
        renamed_docs_dir = sample_docs_dir_func_scope.parent / "renamed_docs"
        os.rename(sample_docs_dir_func_scope, renamed_docs_dir)

        index_after_rename = run_simgrep_command(["index", "--project", "default"])
        assert index_after_rename.exit_code == 0
        cleaned_stdout = index_after_rename.stdout.replace("\n", "").replace("\r", "")
        expected_warning = f"Warning: Path '{sample_docs_dir_func_scope}' does not exist. Skipping."
        assert expected_warning in cleaned_stdout

        search_after_reindex = run_simgrep_command(
            [
                "search",
                "unique_fruit_kiwi",
                "--project",
                "default",
                "--min-score",
                "0.5",
            ]
        )
        assert search_after_reindex.exit_code == 0
        assert "No relevant chunks found" in search_after_reindex.stdout


@pytest.mark.slow
class TestCliConfigE2E:
    def test_command_fails_without_global_config(self, temp_simgrep_home: pathlib.Path) -> None:
        result = run_simgrep_command(["search", "test"])
        assert result.exit_code == 1
        assert "Global config not found" in result.stdout

    def test_search_fails_on_project_config_load_error(self, temp_simgrep_home: pathlib.Path) -> None:
        """Test the SimgrepConfigError handling block in the search command."""
        # Arrange
        with patch(
            "simgrep.main.load_global_config",
            side_effect=SimgrepConfigError("mocked config error"),
        ):
            # Act
            result = run_simgrep_command(["search", "query"])

        # Assert
        assert result.exit_code == 1
        assert "Error during persistent search: mocked config error" in result.stdout

    def test_search_fails_without_index(self, temp_simgrep_home: pathlib.Path) -> None:
        run_simgrep_command(["init", "--global"])
        result = run_simgrep_command(["search", "test"])
        assert result.exit_code == 1
        assert "Persistent index for project 'default' not found" in result.stdout
        assert "Please run 'simgrep index" in result.stdout

    def test_status_on_fresh_init(self, temp_simgrep_home: pathlib.Path) -> None:
        run_simgrep_command(["init", "--global"])
        status_result = run_simgrep_command(["status"])
        assert status_result.exit_code == 0
        assert "Project 'default': 0 files indexed, 0 chunks (index not found)." in status_result.stdout

    def test_index_prompt_decline_prevents_reindexing(self, populated_persistent_index: None) -> None:
        status_before = run_simgrep_command(["status"])
        assert "3 files indexed" in status_before.stdout

        decline_result = run_simgrep_command(["index", "--rebuild"], input_str="n\n")
        assert decline_result.exit_code == 1
        assert "Aborted" in decline_result.stderr

        status_after = run_simgrep_command(["status"])
        assert status_after.stdout == status_before.stdout

    def test_init_global_decline_overwrite(self, temp_simgrep_home: pathlib.Path) -> None:
        run_simgrep_command(["init", "--global"])
        config_file = temp_simgrep_home / ".config" / "simgrep" / "config.toml"
        original_content = config_file.read_text()

        result = run_simgrep_command(["init", "--global"], input_str="n\n")
        assert result.exit_code == 1
        assert "Overwrite?" in result.stdout
        assert "Aborted" in result.stderr
        assert config_file.read_text() == original_content


@pytest.mark.slow
class TestIndexerRobustnessE2E:
    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="os.symlink requires special privileges on Windows",
    )
    def test_index_follows_symlinks(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        project_dir = tmp_path / "symlink_proj"
        project_dir.mkdir()
        content_dir = tmp_path / "content"
        content_dir.mkdir()
        target_file = content_dir / "target.txt"
        target_file.write_text("content from a symlinked file")
        symlink_file = project_dir / "link.txt"
        os.symlink(target_file, symlink_file)

        run_simgrep_command(["init", "--global"])
        run_simgrep_command(["project", "create", "symlink-test"])
        run_simgrep_command(["project", "add-path", str(project_dir), "--project", "symlink-test"])
        run_simgrep_command(["index", "--project", "symlink-test", "--rebuild", "--yes"])

        search_result = run_simgrep_command(["search", "symlinked file", "--project", "symlink-test"])
        assert search_result.exit_code == 0
        assert str(target_file.resolve()) in search_result.stdout.replace("\n", "")

    def test_index_handles_unreadable_file_gracefully(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        project_dir = tmp_path / "unreadable_proj"
        project_dir.mkdir()
        readable_file = project_dir / "readable.txt"
        readable_file.write_text("this is readable content")
        unreadable_file = project_dir / "unreadable.txt"
        unreadable_file.write_text("cannot read this")
        os.chmod(unreadable_file, 0o000)

        try:
            run_simgrep_command(["init", "--global"])
            run_simgrep_command(["project", "create", "unreadable-test"])
            run_simgrep_command(
                [
                    "project",
                    "add-path",
                    str(project_dir),
                    "--project",
                    "unreadable-test",
                ]
            )
            index_result = run_simgrep_command(["index", "--project", "unreadable-test", "--rebuild", "--yes"])

            assert index_result.exit_code == 0
            assert "Error processing" in index_result.stdout
            assert "unreadable.txt" in index_result.stdout
            assert "1 files processed" in index_result.stdout
            assert "1 errors encountered" in index_result.stdout

            search_result = run_simgrep_command(["search", "readable", "--project", "unreadable-test"])
            assert search_result.exit_code == 0
            assert "readable.txt" in search_result.stdout
        finally:
            os.chmod(unreadable_file, 0o644)

    def test_index_skips_binary_files(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        project_dir = tmp_path / "binary_proj"
        project_dir.mkdir()
        (project_dir / "text.txt").write_text("normal text file")
        (project_dir / "archive.zip").write_bytes(b"PK\x03\x04\x14\x00\x00\x00\x08\x00")

        run_simgrep_command(["init", "--global"])
        run_simgrep_command(["project", "create", "binary-test"])
        run_simgrep_command(["project", "add-path", str(project_dir), "--project", "binary-test"])
        index_result = run_simgrep_command(
            [
                "index",
                "--project",
                "binary-test",
                "--rebuild",
                "--pattern",
                "*.*",
                "--yes",
            ]
        )

        assert index_result.exit_code == 0
        assert "files processed" in index_result.stdout

        search_result = run_simgrep_command(["search", "normal text", "--project", "binary-test"])
        assert search_result.exit_code == 0
        assert "text.txt" in search_result.stdout

    def test_index_no_files_found_with_pattern(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        project_dir = tmp_path / "no_match_pattern"
        project_dir.mkdir()
        (project_dir / "file.txt").write_text("some text")

        run_simgrep_command(["init", "--global"])
        run_simgrep_command(["project", "create", "no-match-proj"])
        run_simgrep_command(["project", "add-path", str(project_dir), "--project", "no-match-proj"])
        index_result = run_simgrep_command(
            [
                "index",
                "--project",
                "no-match-proj",
                "--pattern",
                "*.md",
                "--rebuild",
                "--yes",
            ]
        )

        assert index_result.exit_code == 0
        assert "No files found to index" in index_result.stdout


@pytest.mark.slow
class TestCliFilteringE2E:
    @pytest.fixture
    def populated_filtering_index(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        docs_dir = tmp_path / "filtering_docs"
        docs_dir.mkdir()
        (docs_dir / "strong_match.txt").write_text("This document is all about dependency injection in modern software engineering.")
        (docs_dir / "weak_match.txt").write_text("This file is about software.")
        (docs_dir / "code.py").write_text('async def handle_request():\n    print("processing data")')

        run_simgrep_command(["init", "--global"])
        run_simgrep_command(["project", "create", "filtering-proj"])
        run_simgrep_command(["project", "add-path", str(docs_dir), "--project", "filtering-proj"])
        run_simgrep_command(
            [
                "index",
                "--project",
                "filtering-proj",
                "--rebuild",
                "--pattern",
                "*.txt",
                "--pattern",
                "*.py",
                "--yes",
            ]
        )

    def test_search_with_compound_filters(self, populated_filtering_index: None) -> None:
        args = [
            "search",
            "data handling",
            "--project",
            "filtering-proj",
            "--file-filter",
            "*.py",
            "--keyword",
            "async",
            "--output",
            "paths",
        ]
        result = run_simgrep_command(args)
        assert result.exit_code == 0
        assert "code.py" in result.stdout

    def test_search_with_min_score_filters_results(self, populated_filtering_index: None) -> None:
        query = "dependency injection"
        args_low_score = [
            "search",
            query,
            "--project",
            "filtering-proj",
            "--min-score",
            "0.1",
            "--output",
            "paths",
        ]
        result_low_score = run_simgrep_command(args_low_score)
        assert result_low_score.exit_code == 0
        assert "strong_match.txt" in result_low_score.stdout
        assert "weak_match.txt" in result_low_score.stdout

        args_high_score = [
            "search",
            query,
            "--project",
            "filtering-proj",
            "--min-score",
            "0.6",
            "--output",
            "paths",
        ]
        result_high_score = run_simgrep_command(args_high_score)
        assert result_high_score.exit_code == 0
        assert "strong_match.txt" in result_high_score.stdout
        assert "weak_match.txt" not in result_high_score.stdout
