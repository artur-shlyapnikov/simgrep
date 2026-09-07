import json
import os
import pathlib
import sys
from typing import Dict, Generator, List, Optional, Sequence

import pytest
from rich.console import Console
from typer.testing import CliRunner, Result

from simgrep.main import app
from simgrep.models import AppConfig, ProjectConfig
from tests.conftest import FakeRuntime

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

console = Console()
try:
    runner = CliRunner(mix_stderr=False)  # type: ignore[call-arg]
except TypeError:
    runner = CliRunner()


def run_simgrep_command(
    args: List[str],
    cwd: Optional[pathlib.Path] = None,
    env: Optional[Dict[str, str]] = None,
    input_str: Optional[str] = None,
) -> Result:
    """Helper function to run simgrep CLI commands in-process using CliRunner."""
    command = f"simgrep {' '.join(args)}"

    # Set a wide terminal for consistent output in tests, preventing line wrapping.
    e2e_env = env.copy() if env else {}
    e2e_env.setdefault("COLUMNS", "200")

    original_cwd = pathlib.Path.cwd()
    try:
        if cwd:
            os.chdir(cwd)
        result = runner.invoke(app, args, input=input_str, env=e2e_env)
    finally:
        os.chdir(original_cwd)

    if result.exit_code != 0:
        console.print(f"\n[dim]Command failed: {command}[/dim]")
        if cwd:
            console.print(f"[dim]CWD: {cwd}[/dim]")
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


def assert_success(result: Result) -> None:
    assert result.exit_code == 0


def assert_failure_contains(result: Result, phrases: Sequence[str]) -> None:
    assert result.exit_code != 0
    output = f"{result.stdout}\n{result.stderr}".lower()
    for phrase in phrases:
        assert phrase.lower() in output


def assert_clean_json_list(result: Result) -> list[dict]:
    assert_success(result)
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    return payload


def assert_clean_jsonl(result: Result) -> list[dict]:
    assert_success(result)
    assert result.stderr == ""
    rows: list[dict] = []
    for line in result.stdout.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def assert_paths_only(result: Result) -> list[str]:
    assert_success(result)
    assert result.stderr == ""
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert lines
    for line in lines:
        assert "score:" not in line.lower()
        assert "file:" not in line.lower()
    return lines


@pytest.fixture(autouse=True)
def fake_runtime_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime()

    class _Factory:
        def for_app(self, config: AppConfig) -> FakeRuntime:
            del config
            return runtime

        def for_project(self, config: ProjectConfig) -> FakeRuntime:
            del config
            return runtime

    monkeypatch.setattr("simgrep.execution.RuntimeFactory", _Factory)
