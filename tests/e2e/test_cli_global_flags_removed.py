"""E2E coverage: dead global flags (--quiet/-q, --verbose, --color, --no-progress) were removed from the CLI.

They used to be declared on the root callback but were never read anywhere,
so they are misleading CLI surface and must be rejected as usage errors.
"""

from pathlib import Path
from typing import List

import pytest

from tests.e2e.conftest import run_simgrep_command

REMOVED_FLAG_INVOCATIONS: List[List[str]] = [
    ["--quiet"],
    ["-q"],
    ["--verbose"],
    ["--color", "never"],
    ["--color", "NEVER"],
    ["--no-progress"],
]


@pytest.mark.parametrize("flag_args", REMOVED_FLAG_INVOCATIONS)
def test_removed_global_flags_are_rejected_as_usage_errors(flag_args: List[str]) -> None:
    result = run_simgrep_command([*flag_args, "--version"])
    assert result.exit_code == 2
    assert "No such option" in result.stderr


def test_version_flag_still_works() -> None:
    # bare `simgrep --version` must work on its own (no trailing subcommand).
    result = run_simgrep_command(["--version", "models", "status"])
    assert result.exit_code == 0
    assert "simgrep version" in result.stdout


def test_bare_version_flag_works() -> None:
    result = run_simgrep_command(["--version"])
    assert result.exit_code == 0
    assert "simgrep version" in result.stdout


def test_option_only_invocation_prints_help_and_exits_zero(tmp_path: Path) -> None:
    result = run_simgrep_command(["-C", str(tmp_path)])
    assert result.exit_code == 0
    assert "Usage" in result.stdout


@pytest.mark.parametrize("flag_args", [["index", "--quiet"], ["index", "--no-progress"]])
def test_removed_flags_rejected_in_subcommand_position(flag_args: List[str]) -> None:
    """Boundary: removal must hold in every parse position, not just before the subcommand."""
    result = run_simgrep_command(flag_args)
    assert result.exit_code == 2
    assert "No such option" in result.stderr


def test_bare_version_is_usage_error_characterization() -> None:
    """CHARACTERIZATION: bare `simgrep --version` used to be a usage error under
    click 8.x ("Missing command." raised during group parsing, before the eager
    --version callback option could fire). That was fixed fix-first (commit
    92132b3), so this now pins the corrected contract: banner on stdout, exit 0.
    """
    result = run_simgrep_command(["--version"])
    assert result.exit_code == 0
    assert "simgrep version" in result.stdout
