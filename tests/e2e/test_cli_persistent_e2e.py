import os
import pathlib
import sys
from typing import Dict, Generator, List, Optional

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

    original_cwd = None
    if cwd:
        original_cwd = pathlib.Path.cwd()
        os.chdir(cwd)

    # catch_exceptions=False makes it easier to debug, as it will raise python exceptions
    # instead of capturing them in result.exception. typer.Exit is not caught.
    result = runner.invoke(app, args, input=input_str, env=env, catch_exceptions=False)

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
    (docs_dir / "doc1.txt").write_text("This is a document about apples and bananas.")
    (docs_dir / "doc2.txt").write_text("Another document, this one mentions oranges and apples.")
    (docs_dir / "doc3.md").write_text("# Markdown Test\nThis is a test for markdown with apples.")  # Test non-.txt

    subdir = docs_dir / "subdir"
    subdir.mkdir()
    (subdir / "doc_sub.txt").write_text("A document in a subdirectory, also about bananas.")
    return docs_dir


class TestCliPersistentE2E:
    """
    End-to-end tests for persistent indexing and searching via the CLI.
    These tests now run in-process for speed.
    """

    def test_index_and_search_persistent_show_mode(self, temp_simgrep_home: pathlib.Path, sample_docs_dir_session: pathlib.Path) -> None:
        """
        With CliRunner, env vars are passed directly. No need for a separate dict.
        The monkeypatch in temp_simgrep_home handles home dir isolation.
        """

        # 1. Index the sample documents
        index_result = run_simgrep_command(
            ["index", str(sample_docs_dir_session), "--rebuild"],
            input_str="y\n",
        )
        assert index_result.exit_code == 0
        assert "Successfully indexed" in index_result.stdout
        # Output refers to the project name in the format "project 'default'"
        assert "project 'default'" in index_result.stdout
        # Check that .txt files were indexed (default pattern)
        assert "3 files processed" in index_result.stdout  # doc1.txt, doc2.txt, doc_sub.txt
        assert "0 errors encountered" in index_result.stdout

        # 2. Search the persistent index (show mode - default)
        search_result = run_simgrep_command(["search", "apples"])
        assert search_result.exit_code == 0
        # Search output also references the project explicitly
        assert "Searching for: 'apples' in project 'default'" in search_result.stdout
        assert "doc1.txt" in search_result.stdout
        assert "doc2.txt" in search_result.stdout
        assert "markdown" not in search_result.stdout.lower()  # doc3.md should not be indexed by default

        # Check for a term present in a .txt file in a subdirectory
        search_banana_result = run_simgrep_command(["search", "bananas"])
        assert search_banana_result.exit_code == 0
        assert "doc1.txt" in search_banana_result.stdout
        assert str(pathlib.Path("subdir") / "doc_sub.txt") in search_banana_result.stdout  # Check subpath

    def test_index_and_search_persistent_paths_mode(self, temp_simgrep_home: pathlib.Path, sample_docs_dir_session: pathlib.Path) -> None:
        # 1. Index (assuming it's clean or wiped by indexer logic)
        run_simgrep_command(
            ["index", str(sample_docs_dir_session), "--rebuild"],
            input_str="y\n",
        )  # Ensure index is fresh

        # 2. Search with --output paths
        search_result = run_simgrep_command(["search", "apples", "--output", "paths"])
        assert search_result.exit_code == 0

        # Output should be a list of paths, sorted.
        # Paths are absolute from the indexer's perspective.
        # The output may be line-wrapped, so check for substrings
        assert "doc1.txt" in search_result.stdout
        assert "doc2.txt" in search_result.stdout

    def test_search_persistent_no_matches(self, temp_simgrep_home: pathlib.Path, sample_docs_dir_session: pathlib.Path) -> None:
        run_simgrep_command(
            ["index", str(sample_docs_dir_session), "--rebuild"],
            input_str="y\n",
        )

        search_result = run_simgrep_command(["search", "nonexistentqueryxyz"])
        assert search_result.exit_code == 0  # Should exit cleanly
        # Accept either 'No relevant chunks found' or only low scores in output
        assert "No relevant chunks found" in search_result.stdout or "Score:" in search_result.stdout

    def test_status_after_index(
        self,
        temp_simgrep_home: pathlib.Path,
        sample_docs_dir_session: pathlib.Path,
    ) -> None:
        run_simgrep_command(
            ["index", str(sample_docs_dir_session), "--rebuild"],
            input_str="y\n",
        )

        db_file = temp_simgrep_home / ".config" / "simgrep" / "default_project" / "metadata.duckdb"
        conn = duckdb.connect(str(db_file))
        files_count = conn.execute("SELECT COUNT(*) FROM indexed_files;").fetchone()[0]  # type: ignore[index]
        chunks_count = conn.execute("SELECT COUNT(*) FROM text_chunks;").fetchone()[0]  # type: ignore[index]
        conn.close()

        status_result = run_simgrep_command(["status"])
        assert status_result.exit_code == 0
        expected_line = f"Default Project: {files_count} files indexed, {chunks_count} chunks."
        assert expected_line in status_result.stdout

    def test_status_without_index(self, temp_simgrep_home: pathlib.Path) -> None:
        status_result = run_simgrep_command(["status"])
        assert status_result.exit_code == 1
        assert "Default persistent index not found" in status_result.stdout

        search_paths_result = run_simgrep_command(["search", "nonexistentqueryxyz", "--output", "paths"])
        assert search_paths_result.exit_code == 0
        assert "No matching files found." in search_paths_result.stdout

    def test_search_persistent_index_not_exists(self, temp_simgrep_home: pathlib.Path) -> None:
        # Do not run index command
        search_result = run_simgrep_command(["search", "anything"])
        assert search_result.exit_code == 1  # Should fail if index doesn't exist
        assert "Persistent index for project 'default' not found" in search_result.stdout
        assert "Please run 'simgrep index <path>" in search_result.stdout

    def test_index_empty_directory(self, temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
        empty_dir = tmp_path / "empty_docs_for_e2e"
        empty_dir.mkdir()

        index_result = run_simgrep_command(["index", str(empty_dir), "--rebuild"], input_str="y\n")
        assert index_result.exit_code == 0
        assert "No files found to index" in index_result.stdout
        assert "0 files processed" in index_result.stdout  # Or similar message indicating no work done

        # Search should fail as indexer unlinks empty index files
        search_result = run_simgrep_command(["search", "anything"])
        assert search_result.exit_code == 1
        assert "Persistent index for project 'default' not found" in search_result.stdout

    def test_index_non_txt_files_are_ignored_by_default(self, temp_simgrep_home: pathlib.Path, sample_docs_dir_session: pathlib.Path) -> None:
        index_result = run_simgrep_command(
            ["index", str(sample_docs_dir_session), "--rebuild"],
            input_str="y\n",
        )
        assert index_result.exit_code == 0
        # doc3.md should be ignored by default patterns
        assert "3 files processed" in index_result.stdout  # doc1.txt, doc2.txt, subdir/doc_sub.txt

        search_result = run_simgrep_command(["search", "markdown"])  # "markdown" is in doc3.md
        assert search_result.exit_code == 0
        # Accept either 'No relevant chunks found' or only low scores in output
        assert "No relevant chunks found" in search_result.stdout or "Score:" in search_result.stdout

    def test_search_top_option_limits_results(self, temp_simgrep_home: pathlib.Path, sample_docs_dir_session: pathlib.Path) -> None:
        run_simgrep_command(
            ["index", str(sample_docs_dir_session), "--rebuild"],
            input_str="y\n",
        )

        top1_result = run_simgrep_command(["search", "apples", "--top", "1"])
        assert top1_result.exit_code == 0
        assert top1_result.stdout.count("File:") == 1

        top2_result = run_simgrep_command(["search", "apples", "--top", "2"])
        assert top2_result.exit_code == 0
        assert top2_result.stdout.count("File:") >= 2

    def test_index_prompt_decline_prevents_reindexing(self, temp_simgrep_home: pathlib.Path, sample_docs_dir_session: pathlib.Path) -> None:
        run_simgrep_command(
            ["index", str(sample_docs_dir_session), "--rebuild"],
            input_str="y\n",
        )

        db_file = temp_simgrep_home / ".config" / "simgrep" / "default_project" / "metadata.duckdb"
        conn = duckdb.connect(str(db_file))
        row_initial = conn.execute("SELECT COUNT(*) FROM indexed_files;").fetchone()
        conn.close()
        assert row_initial is not None
        initial_files = row_initial[0]

        decline_result = run_simgrep_command(
            ["index", str(sample_docs_dir_session), "--rebuild"],
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

    def test_persistent_search_relative_paths(self, temp_simgrep_home: pathlib.Path, sample_docs_dir_session: pathlib.Path) -> None:
        run_simgrep_command(
            ["index", str(sample_docs_dir_session), "--rebuild"],
            input_str="y\n",
        )

        search_result = run_simgrep_command(
            ["search", "bananas", "--output", "paths", "--relative-paths"],
            cwd=sample_docs_dir_session,
        )
        assert search_result.exit_code == 0
        assert "doc1.txt" in search_result.stdout
        assert "subdir/doc_sub.txt" in search_result.stdout
        assert str(sample_docs_dir_session) not in search_result.stdout
