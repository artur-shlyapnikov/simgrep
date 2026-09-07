import pathlib

from .conftest import assert_clean_json_list, assert_failure_contains, assert_success, run_simgrep_command


def test_machine_formats_keep_stderr_clean(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    docs_dir = tmp_path / "machine_docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("machine payload only", encoding="utf-8")

    for fmt in ("json", "jsonl", "paths", "count", "grep"):
        result = run_simgrep_command(["search", "payload", str(docs_dir), "--format", fmt])
        assert_success(result)
        assert result.stderr == ""


def test_doctor_config_and_reset_commands(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    project_dir = tmp_path / "doctor_proj"
    project_dir.mkdir()
    (project_dir / "a.txt").write_text("reset content", encoding="utf-8")

    assert_success(run_simgrep_command(["init"], cwd=project_dir))
    assert_success(run_simgrep_command(["index", "--rebuild"], cwd=project_dir))

    doctor = run_simgrep_command(["doctor"], cwd=project_dir)
    assert_success(doctor)
    assert "config: ok" in doctor.stdout

    get_before = run_simgrep_command(["config", "get", "lexical_top"], cwd=project_dir)
    assert_success(get_before)
    assert "50" in get_before.stdout

    assert_success(run_simgrep_command(["config", "set", "lexical_top", "7"], cwd=project_dir))
    get_after = run_simgrep_command(["config", "get", "lexical_top"], cwd=project_dir)
    assert_success(get_after)
    assert "7" in get_after.stdout

    assert_success(run_simgrep_command(["reset", "--yes"], cwd=project_dir))
    assert not (project_dir / ".simgrep" / "metadata.duckdb").exists()
    assert not (project_dir / ".simgrep" / "vectors.usearch").exists()


def test_persistent_without_project_fails(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    docs_dir = tmp_path / "non_project"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("hello", encoding="utf-8")

    result = run_simgrep_command(["search", "hello", str(docs_dir), "--persistent", "--format", "json"])
    assert_failure_contains(result, ["persistent", "project"])


def test_persistent_path_scope_uses_indexed_data(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    project_dir = tmp_path / "scope_project"
    docs_dir = project_dir / "docs"
    project_dir.mkdir()
    docs_dir.mkdir()
    file_path = docs_dir / "a.txt"
    file_path.write_text("persistent_scope_term", encoding="utf-8")

    assert_success(run_simgrep_command(["init"], cwd=project_dir))
    assert_success(run_simgrep_command(["project", "add-path", str(docs_dir)], cwd=project_dir))
    assert_success(run_simgrep_command(["index", "--rebuild"], cwd=project_dir))
    file_path.unlink()

    result = run_simgrep_command(["search", "persistent_scope_term", str(docs_dir), "--freshness", "skip", "--format", "json"], cwd=project_dir)
    payload = assert_clean_json_list(result)
    assert payload
