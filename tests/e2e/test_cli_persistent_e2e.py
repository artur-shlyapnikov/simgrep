import json
import os
import pathlib
import shutil
import sys
from typing import Callable, Dict, Generator, List, Optional

import duckdb
import pytest
from rich.console import Console
from typer.testing import CliRunner, Result

from simgrep.main import app

pytest.importorskip("sentence_transformers")
pytest.importorskip("usearch.index")

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

console = Console()
runner = CliRunner(mix_stderr=False)


def run_simgrep_command(
    args: List[str],
    cwd: Optional[pathlib.Path] = None,
    env: Optional[Dict[str, str]] = None,
    input_str: Optional[str] = None,
) -> Result:
    """Helper function to run simgrep CLI commands in-process using CliRunner."""
    command = f"simgrep {' '.join(args)}"
    console.print(f"\n[dim]Running command: {command}[/dim]")
    if cwd:
        console.print(f"[dim]CWD: {cwd}[/dim]")
    if env:
        console.print(f"[dim]Env overrides: {env}[/dim]")

    # Set a wide terminal for consistent output in tests, preventing line wrapping.
    e2e_env = env.copy() if env else {}
    e2e_env.setdefault("COLUMNS", "200")

    original_cwd = None
    if cwd:
        original_cwd = pathlib.Path.cwd()
        os.chdir(cwd)

    result = runner.invoke(app, args, input=input_str, env=e2e_env)

    if original_cwd:
        os.chdir(original_cwd)

    if result.stdout:
        console.print("[bold green]Stdout:[/bold green]")
        console.print(result.stdout)
    if result.stderr:
        console.print("[bold red]Stderr:[/bold red]")
        console.print(result.stderr)
    return result


@pytest.fixture
def temp_simgrep_home(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Generator[pathlib.Path, None, None]:
    """
    Creates a temporary home directory for simgrep E2E tests.
    This fixture ensures that simgrep's configuration and data (like default project)
    are isolated within the test's temporary directory.
    It also monkeypatches os.path.expanduser for the test process itself.
    """
    home_dir = tmp_path / "simgrep_e2e_home"
    home_dir.mkdir(exist_ok=True)

    original_expanduser = os.path.expanduser

    def mock_expanduser_e2e(path_str: str) -> str:
        if path_str == "~" or path_str.startswith("~/"):
            return path_str.replace("~", str(home_dir), 1)
        return original_expanduser(path_str)

    monkeypatch.setattr(os.path, "expanduser", mock_expanduser_e2e)

    yield home_dir


@pytest.fixture(scope="session")
def sample_docs_dir_session(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Creates a sample documents directory for the test session."""
    docs_dir = pathlib.Path(tmp_path_factory.mktemp("sample_docs_e2e"))
    (docs_dir / "doc1.txt").write_text("This is a document about apples and bananas. It also mentions a unique_fruit_kiwi.")
    (docs_dir / "doc2.txt").write_text("Another document, this one mentions oranges and apples.")
    (docs_dir / "doc3.md").write_text("# Markdown Test\nThis is a test for markdown with apples.")  # Test non-.txt

    subdir = docs_dir / "subdir"
    subdir.mkdir()
    (subdir / "doc_sub.txt").write_text("A document in a subdirectory, also about bananas.")
    return docs_dir


def _validate_json_output(result: Result) -> bool:
    try:
        json_output = json.loads(result.stdout)
        assert isinstance(json_output, list)
        assert len(json_output) > 0

        first_result = json_output[0]
        assert "file_path" in first_result
        assert "chunk_text" in first_result
        assert "score" in first_result
        assert "usearch_label" in first_result
        assert "doc" in first_result["file_path"]
        assert "apples" in first_result["chunk_text"].lower()
        return True
    except (json.JSONDecodeError, AssertionError):
        return False


@pytest.fixture
def sample_docs_dir_func_scope(sample_docs_dir_session: pathlib.Path, tmp_path: pathlib.Path) -> pathlib.Path:
    """
    Provides a function-scoped copy of the session-scoped sample docs directory,
    so that tests can modify its contents without affecting other tests.
    """
    dest_dir = tmp_path / "sample_docs_func"
    shutil.copytree(sample_docs_dir_session, dest_dir)
    return dest_dir


@pytest.fixture
def populated_persistent_index_func_scope(temp_simgrep_home: pathlib.Path, sample_docs_dir_func_scope: pathlib.Path) -> None:
    """Creates a default project and indexes the sample documents into it (function scope)."""
    # 0. Global init
    init_result = run_simgrep_command(["init", "--global"])
    assert init_result.exit_code == 0

    # 1. Add path to default project
    add_path_result = run_simgrep_command(["project", "add-path", str(sample_docs_dir_func_scope), "--project", "default"])
    assert add_path_result.exit_code == 0

    # 2. Index the sample documents
    index_result = run_simgrep_command(
        ["index", "--project", "default", "--rebuild", "--yes"],
    )
    assert index_result.exit_code == 0


@pytest.fixture
def populated_persistent_index(temp_simgrep_home: pathlib.Path, sample_docs_dir_session: pathlib.Path) -> None:
    """Creates a default project and indexes the sample documents into it."""
    # 0. Global init
    init_result = run_simgrep_command(["init", "--global"])
    assert init_result.exit_code == 0
    assert "Global simgrep configuration initialized" in init_result.stdout

    # 1. Add path to default project
    add_path_result = run_simgrep_command(["project", "add-path", str(sample_docs_dir_session), "--project", "default"])
    assert add_path_result.exit_code == 0

    # 2. Index the sample documents
    index_result = run_simgrep_command(
        ["index", "--project", "default", "--rebuild", "--yes"],
    )
    assert index_result.exit_code == 0
    assert "Successfully indexed" in index_result.stdout
    assert "project 'default'" in index_result.stdout
    # Check that .txt files were indexed (default pattern)
    assert "3 files processed" in index_result.stdout  # doc1.txt, doc2.txt, doc_sub.txt
    assert "0 errors encountered" in index_result.stdout


class TestCliPersistentE2E:
    """
    End-to-end tests for persistent indexing and searching via the CLI.
    These tests now run in-process for speed.
    """

    @pytest.mark.parametrize(
        "output_mode, extra_args, validation_fn",
        [
            pytest.param(
                "show",
                [],
                lambda r: "File:" in r.stdout and "doc1.txt" in r.stdout and "doc2.txt" in r.stdout,
                id="show_mode",
            ),
            pytest.param(
                "paths",
                [],
                lambda r: "doc1.txt" in r.stdout and "doc2.txt" in r.stdout,
                id="paths_mode",
            ),
            pytest.param("json", [], _validate_json_output, id="json_mode"),
            pytest.param(
                "count",
                ["--min-score", "0.4"],
                lambda r: "2 matching chunks in 2 files" in r.stdout,
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
        """Tests various output modes for persistent search."""
        args = ["search", "apples", "--output", output_mode] + extra_args
        result = run_simgrep_command(args)
        assert result.exit_code == 0
        assert validation_fn(result)

    def test_search_subdirectories(self, populated_persistent_index: None) -> None:
        """Tests that subdirectories are indexed correctly."""
        # Search for 'bananas' which is also in a subdirectory
        search_banana_result = run_simgrep_command(["search", "bananas"])
        assert search_banana_result.exit_code == 0
        clean_banana_stdout = search_banana_result.stdout.replace("\n", "").replace("\r", "")
        assert "doc1.txt" in clean_banana_stdout
        assert os.path.join("subdir", "doc_sub.txt") in clean_banana_stdout

    def test_search_persistent_no_matches(self, populated_persistent_index: None) -> None:
        search_result = run_simgrep_command(["search", "nonexistentqueryxyz"])
        assert search_result.exit_code == 0
        # Accept either 'No relevant chunks found' or only low scores in output
        assert "No relevant chunks found" in search_result.stdout or "Score:" in search_result.stdout

    def test_status_after_index(self, populated_persistent_index: None, temp_simgrep_home: pathlib.Path) -> None:
        db_file = temp_simgrep_home / ".config" / "simgrep" / "default_project" / "metadata.duckdb"
        conn = duckdb.connect(str(db_file))
        files_count = conn.execute("SELECT COUNT(*) FROM indexed_files;").fetchone()[0]  # type: ignore[index]
        chunks_count = conn.execute("SELECT COUNT(*) FROM text_chunks;").fetchone()[0]  # type: ignore[index]
        conn.close()

        status_result = run_simgrep_command(["status"])
        assert status_result.exit_code == 0
        expected_line = f"Default Project: {files_count} files indexed, {chunks_count} chunks."
        assert expected_line in status_result.stdout

    def test_search_persistent_index_not_exists(self, temp_simgrep_home: pathlib.Path) -> None:
        # Do not run index command, should fail because global config is missing
        search_result = run_simgrep_command(["search", "anything"])
        assert search_result.exit_code == 1
        assert "Global config not found" in search_result.stdout

        # After global init, should fail because index is missing
        run_simgrep_command(["init", "--global"])
        search_result_after_init = run_simgrep_command(["search", "anything"])
        assert search_result_after_init.exit_code == 1
        assert "Persistent index for project 'default' not found" in search_result_after_init.stdout
        assert "Please run 'simgrep index" in search_result_after_init.stdout

    def test_index_empty_directory(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        empty_dir = tmp_path / "empty_docs_for_e2e"
        empty_dir.mkdir()

        run_simgrep_command(["init", "--global"])
        run_simgrep_command(["project", "add-path", str(empty_dir)])
        index_result = run_simgrep_command(["index", "--rebuild", "--yes"])
        assert index_result.exit_code == 0
        assert "No files found to index" in index_result.stdout
        assert "0 files processed" in index_result.stdout  # Or similar message indicating no work done

        # Search should now succeed but find nothing, as an empty index is created.
        search_result = run_simgrep_command(["search", "anything"])
        assert search_result.exit_code == 0
        assert "The persistent vector index is empty" in search_result.stdout
        assert "No relevant chunks found" in search_result.stdout

    def test_index_non_txt_files_are_ignored_by_default(self, populated_persistent_index: None) -> None:
        search_result = run_simgrep_command(["search", "markdown"])
        assert search_result.exit_code == 0
        # Accept either 'No relevant chunks found' or only low scores in output
        assert "No relevant chunks found" in search_result.stdout or "Score:" in search_result.stdout

    @pytest.mark.parametrize(
        "top_k, expected_results",
        [
            (1, 1),
            (2, 2),
            (5, 3),  # Request more than available, all 3 should be returned
        ],
    )
    def test_search_top_option_limits_results(self, populated_persistent_index: None, top_k: int, expected_results: int) -> None:
        """Tests that the --top/--k option correctly limits the number of results."""
        result = run_simgrep_command(["search", "apples", "--top", str(top_k)])
        assert result.exit_code == 0
        # Count occurrences of "Score:", which is a reliable marker for one result in 'show' mode.
        assert result.stdout.count("Score:") == expected_results

    def test_index_prompt_decline_prevents_reindexing(self, populated_persistent_index: None, temp_simgrep_home: pathlib.Path) -> None:
        run_simgrep_command(["init", "--global"])
        db_file = temp_simgrep_home / ".config" / "simgrep" / "default_project" / "metadata.duckdb"
        conn = duckdb.connect(str(db_file))
        row_initial = conn.execute("SELECT COUNT(*) FROM indexed_files;").fetchone()
        conn.close()
        assert row_initial is not None
        initial_files = row_initial[0]

        decline_result = run_simgrep_command(
            ["index", "--rebuild"],
            input_str="n\n",
        )
        assert decline_result.exit_code == 1
        assert "Aborted" in decline_result.stderr

        conn = duckdb.connect(str(db_file))
        row_after = conn.execute("SELECT COUNT(*) FROM indexed_files;").fetchone()
        conn.close()
        assert row_after is not None
        after_files = row_after[0]
        assert after_files == initial_files

    def test_persistent_search_relative_paths(self, populated_persistent_index: None, sample_docs_dir_session: pathlib.Path) -> None:
        search_result = run_simgrep_command(
            ["search", "bananas", "--output", "paths", "--relative-paths"],
            cwd=sample_docs_dir_session,
        )
        assert search_result.exit_code == 0
        assert "doc1.txt" in search_result.stdout
        assert os.path.join("subdir", "doc_sub.txt") in search_result.stdout
        assert str(sample_docs_dir_session) not in search_result.stdout

    def test_project_survives_directory_rename_and_reindex(
        self,
        populated_persistent_index_func_scope: None,
        sample_docs_dir_func_scope: pathlib.Path,
    ) -> None:
        """
        Tests that a project can be searched after its source dir is renamed,
        and that a re-index correctly prunes the old path.
        """
        # Search for content, should work
        search_before_rename = run_simgrep_command(["search", "unique_fruit_kiwi", "--project", "default"])
        assert search_before_rename.exit_code == 0
        assert "unique_fruit_kiwi" in search_before_rename.stdout

        # Rename the directory
        renamed_docs_dir = sample_docs_dir_func_scope.parent / "renamed_docs"
        os.rename(sample_docs_dir_func_scope, renamed_docs_dir)

        # Search again by project name. It should still find the content because the absolute path is stored.
        # The path printed will be the old, now non-existent path.
        search_after_rename = run_simgrep_command(
            [
                "search",
                "unique_fruit_kiwi",
                "--project",
                "default",
                "--min-score",
                "0.5",
            ]
        )
        assert search_after_rename.exit_code == 0
        assert "doc1.txt" in search_after_rename.stdout
        assert str(renamed_docs_dir) not in search_after_rename.stdout

        # A re-index should detect the deleted path and prune it.
        index_after_rename = run_simgrep_command(["index", "--project", "default"])
        assert index_after_rename.exit_code == 0
        # The warning message can be wrapped by rich, so we clean up stdout
        # by removing newlines to check for the expected message.
        cleaned_stdout = index_after_rename.stdout.replace("\n", "").replace("\r", "")
        expected_warning = f"Warning: Path '{sample_docs_dir_func_scope}' does not exist. Skipping."
        assert expected_warning in cleaned_stdout

        # After re-indexing, the search should find nothing as the path is gone.
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

    def test_indexing_multiple_disparate_paths_in_one_project(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        """Tests adding multiple separate directory trees to a single project and searching them."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "doc1.txt").write_text("content from docs")
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "util.txt").write_text("text from src")

        # Global init
        run_simgrep_command(["init", "--global"])

        # Create project
        create_result = run_simgrep_command(["project", "create", "multi-path-proj"])
        assert create_result.exit_code == 0

        # Add paths
        add_docs_result = run_simgrep_command(["project", "add-path", str(docs_dir), "--project", "multi-path-proj"])
        assert add_docs_result.exit_code == 0
        add_src_result = run_simgrep_command(["project", "add-path", str(src_dir), "--project", "multi-path-proj"])
        assert add_src_result.exit_code == 0

        # Check that adding the same path again does nothing and succeeds
        add_src_again_result = run_simgrep_command(["project", "add-path", str(src_dir), "--project", "multi-path-proj"])
        assert add_src_again_result.exit_code == 0

        # Index the project
        index_result = run_simgrep_command(["index", "--project", "multi-path-proj", "--rebuild", "--yes"])
        assert index_result.exit_code == 0
        assert "2 files processed" in index_result.stdout  # doc1.txt and util.txt

        # Search and verify
        search_docs_res = run_simgrep_command(["search", "docs", "--project", "multi-path-proj", "--min-score", "0.5"])
        assert search_docs_res.exit_code == 0
        clean_docs_stdout = search_docs_res.stdout.replace("\n", "").replace("\r", "")
        assert "doc1.txt" in clean_docs_stdout
        assert "util.txt" not in clean_docs_stdout

        search_src_res = run_simgrep_command(["search", "src", "--project", "multi-path-proj", "--min-score", "0.5"])
        assert search_src_res.exit_code == 0
        clean_src_stdout = search_src_res.stdout.replace("\n", "").replace("\r", "")
        assert "util.txt" in clean_src_stdout
        assert "doc1.txt" not in clean_src_stdout

    def test_incremental_index_adds_new_file(
        self,
        populated_persistent_index_func_scope: None,
        sample_docs_dir_func_scope: pathlib.Path,
    ) -> None:
        """
        Tests that incremental indexing (no --rebuild) correctly adds a new file.
        """
        # 1. Search for content that doesn't exist yet
        search_before_result = run_simgrep_command(["search", "newly added content", "--min-score", "0.5"])
        assert search_before_result.exit_code == 0
        assert "No relevant chunks found" in search_before_result.stdout

        # 2. Add a new file
        new_file = sample_docs_dir_func_scope / "new_doc.txt"
        new_file.write_text("A new document about newly added content.")

        # 3. Run index again (incrementally)
        index_result = run_simgrep_command(["index"])  # uses default project
        assert index_result.exit_code == 0
        assert "1 chunks indexed" in index_result.stdout
        assert "Skipped (unchanged)" in index_result.stdout

        # 4. Search for the new content
        search_after_result = run_simgrep_command(["search", "newly added content"])
        assert search_after_result.exit_code == 0
        assert "newly added content" in search_after_result.stdout
        assert "new_doc.txt" in search_after_result.stdout

    def test_incremental_index_prunes_deleted_file(
        self,
        populated_persistent_index_func_scope: None,
        sample_docs_dir_func_scope: pathlib.Path,
    ) -> None:
        """
        Tests that incremental indexing correctly prunes a deleted file.
        """
        file_to_delete = sample_docs_dir_func_scope / "doc1.txt"
        content_to_disappear = "unique_fruit_kiwi"

        # 1. Verify content exists
        search_before_result = run_simgrep_command(["search", content_to_disappear])
        assert search_before_result.exit_code == 0
        assert "doc1.txt" in search_before_result.stdout

        # 2. Delete the file
        file_to_delete.unlink()
        assert not file_to_delete.exists()

        # 3. Run index again (incrementally)
        index_result = run_simgrep_command(["index"])  # uses default project
        assert index_result.exit_code == 0

        # 4. Search for the deleted content, it should not be found
        search_after_result = run_simgrep_command(["search", content_to_disappear, "--min-score", "0.5"])
        assert search_after_result.exit_code == 0
        assert "No relevant chunks found" in search_after_result.stdout

    def test_search_from_project_subdirectory_autodetects_project(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        """
        Tests that running commands from a subdirectory of an initialized project
        correctly autodetects and uses the parent project's context.
        """
        project_root = tmp_path / "my-codebase"
        project_subdir = project_root / "src" / "utils"
        project_subdir.mkdir(parents=True)
        (project_root / "README.md").write_text("Project about elegant systems.")

        # 1. Global init
        run_simgrep_command(["init", "--global"])

        # 2. Project init at root
        init_result = run_simgrep_command(["init"], cwd=project_root)
        assert init_result.exit_code == 0
        assert (project_root / ".simgrep").exists()

        # 3. Add path and index from root
        run_simgrep_command(["project", "add-path", str(project_root)], cwd=project_root)
        index_result = run_simgrep_command(
            ["index", "--rebuild", "--pattern", "*.md", "--yes"],
            cwd=project_root,
        )
        assert index_result.exit_code == 0, index_result.stdout

        # 4. Search from subdirectory
        search_result = run_simgrep_command(["search", "elegant systems"], cwd=project_subdir)
        assert search_result.exit_code == 0
        assert "Detected project 'my-codebase'" in search_result.stdout
        assert "README.md" in search_result.stdout
        assert "elegant systems" in search_result.stdout

    def test_init_in_directory_with_spaces(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        """
        Tests that `simgrep init` works correctly in a directory with spaces in its name.
        """
        project_dir = tmp_path / "My Awesome Project"
        project_dir.mkdir()
        (project_dir / "file.txt").write_text("Some content.")
        expected_project_name = "my-awesome-project"

        # 1. Global init
        run_simgrep_command(["init", "--global"])

        # 2. Local init in the directory with spaces
        init_result = run_simgrep_command(["init"], cwd=project_dir)
        assert init_result.exit_code == 0
        assert f"Initialized simgrep project '{expected_project_name}'" in init_result.stdout

        # 3. Verify .simgrep/config.toml
        simgrep_config = project_dir / ".simgrep" / "config.toml"
        assert simgrep_config.exists()
        assert f'project_name = "{expected_project_name}"' in simgrep_config.read_text()

        # 4. Verify indexing works
        index_result = run_simgrep_command(["index", "--rebuild", "--yes"], cwd=project_dir)
        assert index_result.exit_code == 0, index_result.stdout
        assert f"Successfully indexed project '{expected_project_name}'" in index_result.stdout


@pytest.fixture(scope="session")
def sample_docs_for_filtering(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Creates a sample documents directory for filtering tests."""
    docs_dir = tmp_path_factory.mktemp("filtering_docs_e2e")
    (docs_dir / "strong_match.txt").write_text("This document is all about dependency injection in modern software engineering.")
    (docs_dir / "weak_match.txt").write_text("This file is about software.")
    (docs_dir / "case_test.txt").write_text("Notes on the DATABASE. The database is important. This is a database.")
    (docs_dir / "code.py").write_text('async def handle_request():\n    print("processing data")')
    (docs_dir / "unrelated.yml").write_text("config: value")
    return docs_dir


@pytest.fixture
def populated_filtering_index(temp_simgrep_home: pathlib.Path, sample_docs_for_filtering: pathlib.Path) -> None:
    """Creates a 'filtering-proj' and indexes the sample documents into it."""
    # Global init
    init_result = run_simgrep_command(["init", "--global"])
    assert init_result.exit_code == 0

    # Create project
    create_result = run_simgrep_command(["project", "create", "filtering-proj"])
    assert create_result.exit_code == 0

    # Add path
    add_path_result = run_simgrep_command(
        [
            "project",
            "add-path",
            str(sample_docs_for_filtering),
            "--project",
            "filtering-proj",
        ]
    )
    assert add_path_result.exit_code == 0

    # Index with multiple patterns
    index_result = run_simgrep_command(
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
        ],
    )
    assert index_result.exit_code == 0
    assert "Successfully indexed" in index_result.stdout
    assert "4 files processed" in index_result.stdout  # strong, weak, case, code


class TestCliConfigE2E:
    """End-to-end tests for CLI configuration and usability."""

    def test_init_global_overwrite_prompts_and_resets(self, temp_simgrep_home: pathlib.Path) -> None:
        """Tests that `init --global` prompts for overwrite and resets the config."""
        # 1. Initial global setup
        run_simgrep_command(["init", "--global"])
        config_file = temp_simgrep_home / ".config/simgrep/config.toml"
        assert config_file.exists()
        original_content = config_file.read_text()

        # 2. Modify the config file
        with open(config_file, "a") as f:
            f.write("\n# custom modification\n")
        modified_content = config_file.read_text()
        assert original_content != modified_content

        # 3. Run init --global again, confirming overwrite
        overwrite_result = run_simgrep_command(["init", "--global"], input_str="y\n")
        assert overwrite_result.exit_code == 0
        assert "Overwrite?" in overwrite_result.stdout
        assert "Global simgrep configuration initialized" in overwrite_result.stdout

        # 4. Verify config is reset
        final_content = config_file.read_text()
        assert final_content == original_content

    def test_command_fails_without_global_config(self, temp_simgrep_home: pathlib.Path) -> None:
        """Tests that commands fail gracefully if global config is missing."""
        # Run a command that requires global config
        result = run_simgrep_command(["project", "list"])
        assert result.exit_code == 1
        assert "Global config not found" in result.stdout
        assert "Please run 'simgrep init --global'" in result.stdout

    def test_status_on_fresh_init(self, temp_simgrep_home: pathlib.Path) -> None:
        """Tests the status command on a fresh global init."""
        # 1. Global init
        init_result = run_simgrep_command(["init", "--global"])
        assert init_result.exit_code == 0

        # 2. Run status
        status_result = run_simgrep_command(["status"])
        assert status_result.exit_code == 0
        assert "Default Project: 0 files indexed, 0 chunks." in status_result.stdout


class TestCliIndexerRobustnessE2E:
    """Tests for indexer robustness against filesystem oddities."""

    @pytest.mark.skipif(sys.platform == "win32", reason="os.symlink requires special privileges on Windows")
    def test_index_follows_symlink_to_file(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        """Tests that the indexer follows a symbolic link to a file."""
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
        clean_stdout = search_result.stdout.replace("\n", "").replace("\r", "")
        assert str(target_file.resolve()) in clean_stdout
        assert str(symlink_file) not in search_result.stdout

    @pytest.mark.skipif(sys.platform == "win32", reason="os.symlink requires special privileges on Windows")
    def test_index_follows_symlink_to_directory(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        """Tests that the indexer follows a symbolic link to a directory."""
        project_dir = tmp_path / "symlink_dir_proj"
        project_dir.mkdir()
        target_dir = tmp_path / "target_dir"
        target_dir.mkdir()
        (target_dir / "file_in_target.txt").write_text("content from a symlinked directory")
        symlink_dir = project_dir / "link_dir"
        os.symlink(target_dir, symlink_dir, target_is_directory=True)

        run_simgrep_command(["init", "--global"])
        run_simgrep_command(["project", "create", "symlink-dir-test"])
        run_simgrep_command(["project", "add-path", str(project_dir), "--project", "symlink-dir-test"])
        index_result = run_simgrep_command(["index", "--project", "symlink-dir-test", "--rebuild", "--yes"])
        assert index_result.exit_code == 0

        search_result = run_simgrep_command(["search", "symlinked directory", "--project", "symlink-dir-test"])
        assert search_result.exit_code == 0
        target_file = target_dir / "file_in_target.txt"
        clean_stdout = search_result.stdout.replace("\n", "").replace("\r", "")
        assert str(target_file.resolve()) in clean_stdout

    def test_index_handles_unreadable_file_gracefully(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        """Tests that the indexer skips unreadable files with a warning."""
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
            run_simgrep_command(["project", "add-path", str(project_dir), "--project", "unreadable-test"])
            index_result = run_simgrep_command(["index", "--project", "unreadable-test", "--rebuild", "--yes"])

            assert index_result.exit_code == 0
            assert "Error: I/O error processing" in index_result.stdout
            assert "unreadable.txt" in index_result.stdout
            assert "1 files processed" in index_result.stdout
            assert "1 errors encountered" in index_result.stdout

            search_result = run_simgrep_command(["search", "readable", "--project", "unreadable-test"])
            assert search_result.exit_code == 0
            assert "readable.txt" in search_result.stdout
        finally:
            os.chmod(unreadable_file, 0o644)

    def test_index_skips_binary_files(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        """Tests that the indexer does not crash on binary files."""
        project_dir = tmp_path / "binary_proj"
        project_dir.mkdir()
        (project_dir / "text.txt").write_text("normal text file")
        # A simple fake binary file (e.g., first few bytes of a zip file)
        (project_dir / "archive.zip").write_bytes(b"PK\x03\x04\x14\x00\x00\x00\x08\x00")

        run_simgrep_command(["init", "--global"])
        run_simgrep_command(["project", "create", "binary-test"])
        run_simgrep_command(["project", "add-path", str(project_dir), "--project", "binary-test"])
        index_result = run_simgrep_command(["index", "--project", "binary-test", "--rebuild", "--pattern", "*.*", "--yes"])

        assert index_result.exit_code == 0
        assert "1 files processed" in index_result.stdout
        assert "1 errors encountered" in index_result.stdout

        search_result = run_simgrep_command(["search", "normal text", "--project", "binary-test"])
        assert search_result.exit_code == 0
        assert "text.txt" in search_result.stdout


class TestCliFilteringE2E:
    """
    End-to-end tests for search filtering logic.
    """

    def test_search_with_compound_filters(self, populated_filtering_index: None) -> None:
        """
        Tests chaining --file-filter and --keyword.
        """
        # Search for a concept in the python file, then filter by file type and keyword
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
        assert "case_test.txt" not in result.stdout

        # Same search, but wrong file filter
        args_wrong_filter = [
            "search",
            "data handling",
            "--project",
            "filtering-proj",
            "--file-filter",
            "*.txt",
            "--keyword",
            "async",
            "--output",
            "paths",
        ]
        result_wrong = run_simgrep_command(args_wrong_filter)
        assert result_wrong.exit_code == 0
        assert "No matching files found" in result_wrong.stdout

    def test_search_with_min_score_filters_results(self, populated_filtering_index: None) -> None:
        """
        Tests that --min-score correctly excludes low-confidence results.
        """
        # A search for "dependency injection" should match strong_match.txt strongly,
        # and weak_match.txt weakly.
        query = "dependency injection"

        # Search with a low score threshold, expect multiple files
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
        low_score_files = len(result_low_score.stdout.strip().splitlines())
        assert low_score_files >= 1

        # Search with a high score threshold, expect only the strong match
        args_high_score = [
            "search",
            query,
            "--project",
            "filtering-proj",
            "--min-score",
            "0.6",  # High enough to exclude weak matches
            "--output",
            "paths",
        ]
        result_high_score = run_simgrep_command(args_high_score)
        assert result_high_score.exit_code == 0
        assert "strong_match.txt" in result_high_score.stdout
        assert "weak_match.txt" not in result_high_score.stdout
        high_score_files = len(result_high_score.stdout.strip().splitlines())
        assert high_score_files == 1
        assert high_score_files < low_score_files

    def test_keyword_filter_is_case_insensitive(self, populated_filtering_index: None) -> None:
        """
        Tests that the --keyword filter is case-insensitive.
        """
        base_args = [
            "search",
            "notes",
            "--project",
            "filtering-proj",
            "--output",
            "paths",
        ]

        # Test with lowercase
        result_lower = run_simgrep_command(base_args + ["--keyword", "database"])
        assert result_lower.exit_code == 0
        assert "case_test.txt" in result_lower.stdout

        # Test with uppercase
        result_upper = run_simgrep_command(base_args + ["--keyword", "DATABASE"])
        assert result_upper.exit_code == 0
        assert "case_test.txt" in result_upper.stdout

        # Test with mixed case
        result_mixed = run_simgrep_command(base_args + ["--keyword", "Database"])
        assert result_mixed.exit_code == 0
        assert "case_test.txt" in result_mixed.stdout

    def test_search_with_special_characters_in_query(self, populated_filtering_index: None) -> None:
        """
        Tests that a query with punctuation and quotes is handled correctly.
        """
        query = 'what is "dependency injection"?'
        args = ["search", query, "--project", "filtering-proj", "--output", "show"]
        result = run_simgrep_command(args)
        assert result.exit_code == 0
        assert "dependency injection" in result.stdout
        assert "strong_match.txt" in result.stdout
