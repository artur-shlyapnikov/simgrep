"""E2E coverage: CLI command failures must surface as clean `_fail` messages, never tracebacks."""

from __future__ import annotations

import pathlib

import pytest

from tests.e2e.conftest import run_simgrep_command


def _write_corrupted_config(home_dir: pathlib.Path) -> None:
    config_dir = home_dir / ".config" / "simgrep"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text("not [valid toml ===", encoding="utf-8")


@pytest.mark.parametrize(
    "args",
    [
        ["config", "list"],
        ["config", "get", "model"],
        ["doctor"],
        ["models", "status"],
    ],
    ids=["config-list", "config-get", "doctor", "models-status"],
)
def test_corrupted_global_config_fails_cleanly(
    temp_simgrep_home: pathlib.Path,
    args: list[str],
) -> None:
    _write_corrupted_config(temp_simgrep_home)

    result = run_simgrep_command(args)

    assert result.exit_code == 1
    assert "Error" in result.stderr
    assert "Hint:" in result.stderr
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr


def test_config_set_corrupted_config_fails_cleanly(
    temp_simgrep_home: pathlib.Path,
) -> None:
    _write_corrupted_config(temp_simgrep_home)

    result = run_simgrep_command(["config", "set", "model", "x"])

    assert result.exit_code == 1
    assert "Error" in result.stderr
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr


def test_models_cache_failure_is_clean_error(
    temp_simgrep_home: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", boom)
    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)

    result = run_simgrep_command(["models", "cache", "some-model"])

    assert result.exit_code == 1
    assert "Failed to cache model 'some-model'" in result.stderr
    assert "boom" in result.stderr
    assert "Hint: Check the model id" in result.stderr
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr


def test_models_cache_corrupted_config_without_model_arg_fails_cleanly(
    temp_simgrep_home: pathlib.Path,
) -> None:
    _write_corrupted_config(temp_simgrep_home)

    result = run_simgrep_command(["models", "cache"])

    assert result.exit_code == 1
    assert "Error" in result.stderr
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr


def test_models_status_explicit_model_bypasses_broken_config(
    temp_simgrep_home: pathlib.Path,
) -> None:
    _write_corrupted_config(temp_simgrep_home)

    result = run_simgrep_command(["models", "status", "some-org/some-model"])

    assert result.exit_code == 0
    assert "not cached" in result.stdout


def test_models_cache_programming_error_rewrapped_as_user_error(
    temp_simgrep_home: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise TypeError("'NoneType' object is not subscriptable")

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", boom)

    result = run_simgrep_command(["models", "cache", "some-model"])

    assert result.exit_code == 1
    assert "Failed to cache model 'some-model'" in result.stderr
    assert "'NoneType' object is not subscriptable" in result.stderr
    assert "Hint: Check the model id" in result.stderr
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr


def test_models_cache_success_path_unaffected(
    temp_simgrep_home: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_from_pretrained(selected: str) -> str:
        calls.append("tokenizer")
        return selected

    def fake_snapshot_download(selected: str, **kwargs: object) -> str:
        calls.append("snapshot")
        return selected

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", fake_from_pretrained)
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    result = run_simgrep_command(["models", "cache", "fake-model"])

    assert result.exit_code == 0
    assert "fake-model: cached" in result.stdout
    assert sorted(calls) == ["snapshot", "tokenizer"]  # type: ignore[assignment]


@pytest.mark.parametrize(
    ("args", "phrase"),
    [
        (["config", "get", "nope"], "Unknown config key"),
        (["config", "set", "nope", "3"], "Unknown config key"),
    ],
    ids=["get-unknown-key", "set-unknown-key"],
)
def test_unknown_config_key_fails_cleanly(
    temp_simgrep_home: pathlib.Path,
    args: list[str],
    phrase: str,
) -> None:
    result = run_simgrep_command(args)

    assert result.exit_code == 1
    assert phrase in result.stderr
    assert "Traceback" not in result.stderr


def test_corrupt_index_database_fails_with_recovery_hint(
    temp_simgrep_home: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    project_dir = tmp_path / "corrupt_project"
    project_dir.mkdir()
    (project_dir / "a.py").write_text("x = 1\n", encoding="utf-8")
    assert run_simgrep_command(["init"], cwd=project_dir).exit_code == 0
    db_path = project_dir / ".simgrep" / "metadata.duckdb"
    db_path.write_bytes(b"garbage")

    result = run_simgrep_command(["status"], cwd=project_dir)

    assert result.exit_code == 1
    assert "Failed to open metadata database" in result.stderr
    assert "reset" in result.stderr
    assert "Traceback" not in result.stderr
