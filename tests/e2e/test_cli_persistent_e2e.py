import os
import pathlib
import subprocess
import sys
import time  # For potential debugging delays
from typing import Dict, List, Optional

import pytest
from rich.console import Console

# Add the project root to the Python path for the test runner
# This ensures that 'simgrep' can be imported if tests are run directly
# and not via 'make test' or 'uv run pytest' which might handle PYTHONPATH.
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Assuming conftest.py is in the same directory or a parent directory
# and defines sample_docs_dir. If not, this import might need adjustment
# or the fixture needs to be defined here.
# from .conftest import sample_docs_dir # If conftest is at tests/e2e/conftest.py

console = Console()


def run_simgrep_command(
    args: List[str],
    cwd: Optional[pathlib.Path] = None,
    env: Optional[Dict[str, str]] = None,  # Added env parameter
) -> subprocess.CompletedProcess:
    """Helper function to run simgrep CLI commands."""
    command = [sys.executable, "-m", "simgrep.main"] + args
    # Ensure environment variables are correctly passed
    process_env = os.environ.copy()
    if env:
        process_env.update(env)

    console.print(f"\n[dim]Running command: {' '.join(command)}[/dim]")
    if cwd:
        console.print(f"[dim]CWD: {cwd}[/dim]")
    if env:
        console.print(f"[dim]Env overrides: {env}[/dim]")

    result = subprocess.run(
        command, capture_output=True, text=True, cwd=cwd, env=process_env
    )

    if result.stdout:
        console.print("[bold green]Stdout:[/bold green]")
        console.print(result.stdout)
    if result.stderr:
        console.print("[bold red]Stderr:[/bold red]")
        console.print(result.stderr)
    return result


@pytest.fixture
def temp_simgrep_home(tmp_path: pathlib.Path, monkeypatch) -> pathlib.Path:
    """
    Creates a temporary home directory for simgrep E2E tests.
    This fixture ensures that simgrep's configuration and data (like default project)
    are isolated within the test's temporary directory.
    It also monkeypatches os.path.expanduser for the test process itself,
    and yields the path to be used for the HOME env var in subprocesses.
    """
    home_dir = tmp_path / "simgrep_e2e_home"
    home_dir.mkdir(exist_ok=True)

    # Monkeypatch os.path.expanduser for the test runner's process.
    # This is less critical than setting HOME for the subprocess but ensures consistency
    # if any simgrep code is directly imported and run in the test process.
    original_expanduser = os.path.expanduser

    def mock_expanduser_e2e(path_str: str) -> str:
        if path_str == "~" or path_str.startswith("~/"):
            return path_str.replace("~", str(home_dir), 1)
        return original_expanduser(path_str)

    monkeypatch.setattr(os.path, "expanduser", mock_expanduser_e2e)

    yield home_dir  # Yield the created temporary home directory path

    # Cleanup is handled by tmp_path. Unpatching if necessary:
    # monkeypatch.setattr(os.path, "expanduser", original_expanduser)


@pytest.fixture(scope="session")
def sample_docs_dir_session(tmp_path_factory) -> pathlib.Path:
    """Creates a sample documents directory for the test session."""
    docs_dir = tmp_path_factory.mktemp("sample_docs_e2e")
    (docs_dir / "doc1.txt").write_text("This is a document about apples and bananas.")
    (docs_dir / "doc2.txt").write_text(
        "Another document, this one mentions oranges and apples."
    )
    (docs_dir / "doc3.md").write_text(
        "# Markdown Test\nThis is a test for markdown with apples."
    )  # Test non-.txt

    subdir = docs_dir / "subdir"
    subdir.mkdir()
    (subdir / "doc_sub.txt").write_text(
        "A document in a subdirectory, also about bananas."
    )
    return docs_dir


class TestCliPersistentE2E:
    """
    End-to-end tests for persistent indexing and searching via the CLI.
    These tests interact with the file system and subprocesses.
    """

    def test_index_and_search_persistent_show_mode(
        self, temp_simgrep_home: pathlib.Path, sample_docs_dir_session: pathlib.Path
    ):
        env_vars = {"HOME": str(temp_simgrep_home)}

        # 1. Index the sample documents
        index_result = run_simgrep_command(
            ["index", str(sample_docs_dir_session)], env=env_vars
        )
        assert index_result.returncode == 0
        assert "Successfully indexed" in index_result.stdout
        assert "default project" in index_result.stdout
        # Check that .txt files were indexed (default pattern)
        assert (
            "3 files processed" in index_result.stdout
        )  # doc1.txt, doc2.txt, doc_sub.txt
        assert "0 errors encountered" in index_result.stdout

        # 2. Search the persistent index (show mode - default)
        search_result = run_simgrep_command(["search", "apples"], env=env_vars)
        assert search_result.returncode == 0
        assert (
            "Searching for: 'apples' in default persistent index"
            in search_result.stdout
        )
        assert "doc1.txt" in search_result.stdout
        assert "doc2.txt" in search_result.stdout
        assert (
            "markdown" not in search_result.stdout.lower()
        )  # doc3.md should not be indexed by default

        # Check for a term present in a .txt file in a subdirectory
        search_banana_result = run_simgrep_command(["search", "bananas"], env=env_vars)
        assert search_banana_result.returncode == 0
        assert "doc1.txt" in search_banana_result.stdout
        assert (
            str(pathlib.Path("subdir") / "doc_sub.txt") in search_banana_result.stdout
        )  # Check subpath

    def test_index_and_search_persistent_paths_mode(
        self, temp_simgrep_home: pathlib.Path, sample_docs_dir_session: pathlib.Path
    ):
        env_vars = {"HOME": str(temp_simgrep_home)}

        # 1. Index (assuming it's clean or wiped by indexer logic)
        run_simgrep_command(
            ["index", str(sample_docs_dir_session)], env=env_vars
        )  # Ensure index is fresh

        # 2. Search with --output paths
        search_result = run_simgrep_command(
            ["search", "apples", "--output", "paths"], env=env_vars
        )
        assert search_result.returncode == 0

        # Output should be a list of paths, sorted.
        # Paths are absolute from the indexer's perspective.
        expected_path1 = str((sample_docs_dir_session / "doc1.txt").resolve())
        expected_path2 = str((sample_docs_dir_session / "doc2.txt").resolve())

        # The order in output can vary, so check presence
        assert expected_path1 in search_result.stdout
        assert expected_path2 in search_result.stdout

        # Ensure paths are sorted in the output if multiple are printed
        # This is harder to check directly without parsing stdout carefully.
        # For now, presence is the main check.
        # Example of a more robust check if output is guaranteed sorted:
        # lines = [line.strip() for line in search_result.stdout.splitlines() if line.strip().endswith(".txt")]
        # assert lines == sorted(lines)

    def test_search_persistent_no_matches(
        self, temp_simgrep_home: pathlib.Path, sample_docs_dir_session: pathlib.Path
    ):
        env_vars = {"HOME": str(temp_simgrep_home)}
        run_simgrep_command(["index", str(sample_docs_dir_session)], env=env_vars)

        search_result = run_simgrep_command(
            ["search", "nonexistentqueryxyz"], env=env_vars
        )
        assert search_result.returncode == 0  # Should exit cleanly
        assert "No relevant chunks found" in search_result.stdout  # For show mode

        search_paths_result = run_simgrep_command(
            ["search", "nonexistentqueryxyz", "--output", "paths"], env=env_vars
        )
        assert search_paths_result.returncode == 0
        assert "No matching files found." in search_paths_result.stdout

    def test_search_persistent_index_not_exists(self, temp_simgrep_home: pathlib.Path):
        env_vars = {"HOME": str(temp_simgrep_home)}
        # Do not run index command
        search_result = run_simgrep_command(["search", "anything"], env=env_vars)
        assert search_result.returncode == 1  # Should fail if index doesn't exist
        assert "Default persistent index not found" in search_result.stdout
        assert "Please run 'simgrep index <path>' first" in search_result.stdout

    def test_index_empty_directory(
        self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path
    ):
        env_vars = {"HOME": str(temp_simgrep_home)}
        empty_dir = tmp_path / "empty_docs_for_e2e"
        empty_dir.mkdir()

        index_result = run_simgrep_command(["index", str(empty_dir)], env=env_vars)
        assert index_result.returncode == 0
        assert "No files found to index" in index_result.stdout
        assert (
            "0 files processed" in index_result.stdout
        )  # Or similar message indicating no work done

        # Search should find nothing
        search_result = run_simgrep_command(["search", "anything"], env=env_vars)
        assert search_result.returncode == 0  # Index is empty, but exists
        assert "The persistent vector index is empty" in search_result.stdout
        # Or "No relevant chunks found" depending on exact flow for empty index

    def test_index_non_txt_files_are_ignored_by_default(
        self, temp_simgrep_home: pathlib.Path, sample_docs_dir_session: pathlib.Path
    ):
        env_vars = {"HOME": str(temp_simgrep_home)}

        index_result = run_simgrep_command(
            ["index", str(sample_docs_dir_session)], env=env_vars
        )
        assert index_result.returncode == 0
        # doc3.md should be ignored by default patterns
        assert (
            "3 files processed" in index_result.stdout
        )  # doc1.txt, doc2.txt, subdir/doc_sub.txt

        search_result = run_simgrep_command(
            ["search", "markdown"], env=env_vars
        )  # "markdown" is in doc3.md
        assert search_result.returncode == 0
        assert (
            "No relevant chunks found" in search_result.stdout
        )  # Should not find "markdown"
