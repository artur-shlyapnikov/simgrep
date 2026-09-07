import pathlib

from .conftest import assert_success, run_simgrep_command


def test_local_init_index_search_status_flow(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    project_dir = tmp_path / "local-proj"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("local workflow semantic content")

    assert_success(run_simgrep_command(["init"], cwd=project_dir))
    assert_success(run_simgrep_command(["index", "--rebuild"], cwd=project_dir))

    search = run_simgrep_command(["search", "semantic content"], cwd=project_dir)
    assert_success(search)
    assert "file.txt" in search.stdout

    status = run_simgrep_command(["status"], cwd=project_dir)
    assert_success(status)
    assert "file(s)" in status.stdout


def test_project_add_remove_info_flow(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    project_dir = tmp_path / "proj"
    docs_dir = project_dir / "docs"
    project_dir.mkdir()
    docs_dir.mkdir()
    assert_success(run_simgrep_command(["init"], cwd=project_dir))
    assert_success(run_simgrep_command(["project", "add-path", str(docs_dir)], cwd=project_dir))
    info = run_simgrep_command(["project", "info"], cwd=project_dir)
    assert_success(info)
    assert str(docs_dir) in info.stdout
    assert_success(run_simgrep_command(["project", "remove-path", str(docs_dir)], cwd=project_dir))


def test_project_root_option_runs_commands_from_outside_project(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    project_dir = tmp_path / "external-proj"
    outsider_dir = tmp_path / "outside"
    project_dir.mkdir()
    outsider_dir.mkdir()
    (project_dir / "file.txt").write_text("project root option content", encoding="utf-8")

    assert_success(run_simgrep_command(["init", str(project_dir)]))
    assert_success(run_simgrep_command(["-C", str(project_dir), "index", "--rebuild"], cwd=outsider_dir))

    search = run_simgrep_command(["-C", str(project_dir), "search", "project root option content"], cwd=outsider_dir)
    assert_success(search)
    assert "file.txt" in search.stdout

    status = run_simgrep_command(["-C", str(project_dir), "status"], cwd=outsider_dir)
    assert_success(status)
    assert "file(s)" in status.stdout


def test_project_root_short_flag_and_search_context_short_flag_do_not_conflict(temp_simgrep_home: pathlib.Path, tmp_path: pathlib.Path) -> None:
    project_dir = tmp_path / "proj-flags"
    outsider_dir = tmp_path / "outside-flags"
    project_dir.mkdir()
    outsider_dir.mkdir()
    (project_dir / "file.txt").write_text("line one\nline two content\nline three", encoding="utf-8")

    assert_success(run_simgrep_command(["init", str(project_dir)]))
    assert_success(run_simgrep_command(["-C", str(project_dir), "index", "--rebuild"], cwd=outsider_dir))

    result = run_simgrep_command(
        ["-C", str(project_dir), "search", "content", "--format", "json", "-c", "1"],
        cwd=outsider_dir,
    )
    assert_success(result)
    assert '"line_start"' in result.stdout
