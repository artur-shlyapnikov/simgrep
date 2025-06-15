import json
import os
import pathlib
import shutil
import sys
from typing import Callable, Dict, Generator, List, Optional

import pytest
from rich.console import Console
from typer.testing import CliRunner, Result

from simgrep.main import app

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
        assert "label" in first_result
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
