from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from simgrep.errors import SimgrepError
from simgrep.files import FileScanEntry, build_file_plan, classify_file, is_test_path, scan_files
from simgrep.indexing import IndexEngine
from simgrep.models import SCHEMA_VERSION, AppConfig, ChangeDetectionMode, FileRecord, FileRole, IndexOptions, ProjectConfig, ScanOptions, SearchOptions
from simgrep.search import SearchEngine
from simgrep.store import Store
from tests.conftest import FakeRuntime


def _project(tmp_path: Path, *indexed: Path) -> ProjectConfig:
    paths = indexed if indexed else (tmp_path,)
    return ProjectConfig(SCHEMA_VERSION, "p", tmp_path, tuple(paths), "fake", 128, 20)


def test_scan_files_respects_gitignore_repo_ignore_and_sensitive(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / ".repo_ignore").write_text("repo_ignored.py\n", encoding="utf-8")
    (tmp_path / "ok.py").write_text("print('ok')", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("print('x')", encoding="utf-8")
    (tmp_path / "repo_ignored.py").write_text("print('y')", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
    scanned = scan_files(tmp_path, ScanOptions())
    assert [entry.rel_path for entry in scanned] == ["ok.py"]


def test_build_file_plan_stat_and_hash(tmp_path: Path) -> None:
    file_a = tmp_path / "a.py"
    file_b = tmp_path / "b.py"
    file_a.write_text("alpha", encoding="utf-8")
    file_b.write_text("beta", encoding="utf-8")
    stat_a = file_a.stat()
    stat_b = file_b.stat()
    scanned = [
        FileScanEntry(file_a, file_a.resolve(), "a.py", stat_a.st_size, stat_a.st_mtime_ns),
        FileScanEntry(file_b, file_b.resolve(), "b.py", stat_b.st_size, stat_b.st_mtime_ns),
    ]
    existing = {
        file_a.resolve(): FileRecord(id=1, path=file_a.resolve(), size_bytes=stat_a.st_size, mtime_ns=stat_a.st_mtime_ns, sha256="same"),
        (tmp_path / "gone.py").resolve(): FileRecord(id=2, path=(tmp_path / "gone.py").resolve(), size_bytes=1, mtime_ns=1, sha256="x"),
    }
    plan_stat = build_file_plan(scanned, existing, options=ScanOptions(), change_detection=ChangeDetectionMode.stat)
    statuses_stat = {entry.path.name: entry.status for entry in plan_stat.entries}
    assert statuses_stat["a.py"] == "unchanged"
    assert statuses_stat["b.py"] == "new"
    assert statuses_stat["gone.py"] == "deleted"

    file_a.write_text("alpha changed", encoding="utf-8")
    stat_a2 = file_a.stat()
    plan_hash = build_file_plan(
        [FileScanEntry(file_a, file_a.resolve(), "a.py", stat_a2.st_size, stat_a2.st_mtime_ns)],
        {file_a.resolve(): FileRecord(id=1, path=file_a.resolve(), size_bytes=stat_a.st_size, mtime_ns=stat_a.st_mtime_ns, sha256="same")},
        options=ScanOptions(),
        change_detection=ChangeDetectionMode.hash,
    )
    assert len(plan_hash.entries) == 1
    assert plan_hash.entries[0].status == "changed"


def test_classify_file_roles_for_source_test_docs_config_and_metadata() -> None:
    assert classify_file(Path("src/main/java/com/acme/App.java")).file_role == FileRole.source

    assert classify_file(Path("src/test/java/com/acme/AppTest.java")).file_role == FileRole.test
    assert classify_file(Path("src/integrationTest/java/com/acme/AppIT.java")).file_role == FileRole.test
    assert classify_file(Path("tests/test_api.py")).file_role == FileRole.test
    assert classify_file(Path("pkg/model_test.py")).file_role == FileRole.test
    assert classify_file(Path("web/user.spec.ts")).file_role == FileRole.test

    assert classify_file(Path("README.md")).file_role == FileRole.docs
    assert classify_file(Path("docs/guide.rst")).file_role == FileRole.docs
    assert classify_file(Path("notes/overview.txt")).file_role == FileRole.docs

    assert classify_file(Path("config/app.yaml")).file_role == FileRole.config
    assert classify_file(Path("config/app.json")).file_role == FileRole.config
    assert classify_file(Path("config/app.xml")).file_role == FileRole.config
    assert classify_file(Path("config/app.toml")).file_role == FileRole.config
    assert classify_file(Path("config/app.properties")).file_role == FileRole.config

    assert classify_file(Path("pom.xml")).file_role == FileRole.build_metadata
    assert classify_file(Path("build.gradle")).file_role == FileRole.build_metadata
    assert classify_file(Path("Dockerfile")).file_role == FileRole.build_metadata
    assert classify_file(Path("Makefile")).file_role == FileRole.build_metadata
    assert classify_file(Path("justfile")).file_role == FileRole.build_metadata

    assert classify_file(Path("poetry.lock")).file_role == FileRole.dependency_metadata
    assert classify_file(Path("renovate.json")).file_role == FileRole.dependency_metadata


def test_classify_generated_paths_as_generated_role() -> None:
    assert classify_file(Path("generated/client.py")).file_role == FileRole.generated
    assert classify_file(Path("target/generated-sources/openapi/Api.java")).file_role == FileRole.generated
    assert classify_file(Path("build/generated/source/proto/main.py")).file_role == FileRole.generated


def test_scan_files_supports_nested_gitignore_patterns_globstar_and_negation(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "keep.py").write_text("print('keep')", encoding="utf-8")
    (tmp_path / "src" / "drop.py").write_text("print('drop')", encoding="utf-8")
    (tmp_path / "src" / "nested").mkdir()
    (tmp_path / "src" / "nested" / "drop.py").write_text("print('drop nested')", encoding="utf-8")
    (tmp_path / "src" / "nested" / "keep.py").write_text("print('keep nested')", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("src/**/drop.py\n!src/nested/drop.py\n", encoding="utf-8")

    scanned = scan_files(tmp_path, ScanOptions())
    rel_paths = sorted(entry.rel_path for entry in scanned)
    assert rel_paths == ["src/keep.py", "src/nested/drop.py", "src/nested/keep.py"]


def test_scan_files_supports_directory_ignore_patterns(tmp_path: Path) -> None:
    (tmp_path / "data" / "generated").mkdir(parents=True)
    (tmp_path / "data" / "generated" / "skip.py").write_text("print('skip')", encoding="utf-8")
    (tmp_path / "data" / "keep.py").write_text("print('keep')", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("data/generated/\n", encoding="utf-8")

    scanned = scan_files(tmp_path, ScanOptions())
    assert [entry.rel_path for entry in scanned] == ["data/keep.py"]


def test_scan_files_ignores_reserved_dirs_even_when_not_in_gitignore(tmp_path: Path) -> None:
    for folder in (".simgrep", ".git", "node_modules", "target", ".venv"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "x.py").write_text("print('x')", encoding="utf-8")
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".pytest_cache" / "cache.py").write_text("print('cache')", encoding="utf-8")
    (tmp_path / "ok.py").write_text("print('ok')", encoding="utf-8")

    scanned = scan_files(tmp_path, ScanOptions())
    assert [entry.rel_path for entry in scanned] == ["ok.py"]


@pytest.mark.parametrize("name", [".env", ".env.local", "prod.pem", "private.key", "id_rsa", "id_ed25519", "cluster.kubeconfig"])
def test_scan_direct_sensitive_file_is_never_indexed(tmp_path: Path, name: str) -> None:
    secret = tmp_path / name
    secret.write_text("TOP_SECRET=1", encoding="utf-8")
    assert scan_files(secret, ScanOptions()) == []


def test_sensitive_file_content_not_persisted_or_searchable(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    (tmp_path / ".env").write_text("VERY_SECRET_TOKEN", encoding="utf-8")
    (tmp_path / "ok.py").write_text("public token marker", encoding="utf-8")
    project = _project(tmp_path)
    app = AppConfig(model="fake")

    IndexEngine(fake_runtime).index_project(project, app, IndexOptions(rebuild=True))

    store = Store.open(project.metadata_db_path)
    try:
        files = store.get_files()
        assert list(path.name for path in files) == ["ok.py"]
    finally:
        store.close()

    outcome = SearchEngine(fake_runtime).search_project(project, app, SearchOptions(query="VERY_SECRET_TOKEN"), freshness=app.freshness)
    assert all(result.file_path.name != ".env" and "VERY_SECRET_TOKEN" not in result.chunk_text for result in outcome.results)


def test_max_file_size_marks_too_large_and_does_not_index(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    (tmp_path / "big.py").write_text("0123456789ABCDEF", encoding="utf-8")
    project = _project(tmp_path)
    app = AppConfig(model="fake", max_file_size_bytes=8)

    plan = IndexEngine(fake_runtime).plan_project(project, app)
    assert len(plan.entries) == 1
    assert plan.entries[0].status == "too_large"

    stats = IndexEngine(fake_runtime).index_project(project, app, IndexOptions(rebuild=True))
    assert stats.files_skipped_too_large == 1
    assert stats.files_indexed == 0
    assert stats.chunks_indexed == 0


def test_unreadable_scan_root_raises_simgrep_error(tmp_path: Path) -> None:
    # f9da6e5: unreadable scan root raises instead of silently returning [].
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    real_scandir = os.scandir

    def patched_scandir(path: str | os.PathLike[str]) -> Iterator[os.DirEntry[str]]:
        if Path(path) == tmp_path:
            raise OSError("scan denied")
        return real_scandir(path)

    with patch("os.scandir", patched_scandir):
        with pytest.raises(SimgrepError, match="Cannot read directory"):
            scan_files(tmp_path, ScanOptions())


def test_unreadable_file_in_hash_is_controlled_unreadable_status(tmp_path: Path) -> None:
    file_path = tmp_path / "a.py"
    file_path.write_text("alpha", encoding="utf-8")
    stat = file_path.stat()
    discovered = [FileScanEntry(file_path, file_path.resolve(), "a.py", stat.st_size, stat.st_mtime_ns)]
    existing = {file_path.resolve(): FileRecord(id=1, path=file_path.resolve(), size_bytes=1, mtime_ns=1, sha256="old")}

    with patch("simgrep.files.calculate_file_hash", side_effect=OSError("hash denied")):
        plan = build_file_plan(discovered, existing, options=ScanOptions(), change_detection=ChangeDetectionMode.hash)

    assert len(plan.entries) == 1
    assert plan.entries[0].status == "unreadable"
    assert plan.entries[0].reason == "hash_failed"


def test_binary_content_with_text_extension_is_not_indexed_as_noise(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    (tmp_path / "binary.py").write_bytes(b"\x00\x01\x02\x03\x04")
    project = _project(tmp_path)
    app = AppConfig(model="fake")

    stats = IndexEngine(fake_runtime).index_project(project, app, IndexOptions(rebuild=True))
    assert stats.files_seen == 1
    # Binary content yields zero chunks: recorded as a file row (no chunks, no
    # vectors) so freshness plans report it unchanged instead of new forever.
    assert stats.files_indexed == 1
    assert stats.chunks_indexed == 0
    follow_up = IndexEngine(fake_runtime).plan_project(project, app, IndexOptions())
    assert follow_up.has_mutations is False


def test_unsupported_extension_uses_unstructured_fallback_and_failure_logs_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from simgrep.adapters.extractor import TextExtractor

    target = tmp_path / "doc.weird"
    target.write_text("fallback me", encoding="utf-8")

    extractor = TextExtractor()
    caplog.clear()
    with patch("builtins.__import__", side_effect=ImportError("no unstructured")):
        text = extractor.extract(target)

    assert text == ""
    assert "Failed to extract" in caplog.text


def test_duplicate_indexed_paths_do_not_index_same_file_twice(tmp_path: Path, fake_runtime: FakeRuntime) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("one", encoding="utf-8")
    project = _project(tmp_path, tmp_path, src)
    app = AppConfig(model="fake")

    stats = IndexEngine(fake_runtime).index_project(project, app, IndexOptions(rebuild=True))
    assert stats.files_seen == 1
    assert stats.files_indexed == 1


def test_direct_file_path_scan_uses_parent_as_base_and_only_that_file(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.py").write_text("beta", encoding="utf-8")

    scanned = scan_files(tmp_path / "a.py", ScanOptions())
    assert len(scanned) == 1
    assert scanned[0].rel_path == "a.py"
    assert scanned[0].path.name == "a.py"


def test_symlink_not_followed_by_default(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real_file = real_dir / "a.py"
    real_file.write_text("alpha", encoding="utf-8")
    (tmp_path / "link").symlink_to(real_dir, target_is_directory=True)

    scanned = scan_files(tmp_path, ScanOptions())
    assert [(entry.path, entry.resolved_path) for entry in scanned] == [(real_file, real_file.resolve())]


def test_symlink_sorting_before_real_dir_does_not_hide_real_dir(tmp_path: Path) -> None:
    b_real = tmp_path / "b_real"
    b_real.mkdir()
    mod_file = b_real / "mod.py"
    mod_file.write_text("module", encoding="utf-8")
    (tmp_path / "a_link").symlink_to(b_real, target_is_directory=True)

    scanned = scan_files(tmp_path, ScanOptions())
    assert [(entry.path, entry.resolved_path) for entry in scanned] == [(mod_file, mod_file.resolve())]


def test_follow_symlinks_indexes_target_once_and_avoids_cycle(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "a.py").write_text("alpha", encoding="utf-8")
    (tmp_path / "link").symlink_to(real_dir, target_is_directory=True)
    (real_dir / "back").symlink_to(tmp_path, target_is_directory=True)

    scanned = scan_files(tmp_path, ScanOptions(follow_symlinks=True))
    assert [entry.resolved_path.name for entry in scanned] == ["a.py"]


def test_follow_symlinks_skips_outside_root_target(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-for-simgrep-test.py"
    outside.write_text("outside", encoding="utf-8")
    try:
        (tmp_path / "outside.py").symlink_to(outside)
        scanned = scan_files(tmp_path, ScanOptions(follow_symlinks=True))
        assert scanned == []
    finally:
        if outside.exists():
            outside.unlink()


def test_scan_files_globs_filter_and_single_file_inside_ignored_dir(tmp_path: Path) -> None:
    (tmp_path / "keep.md").write_text("a")
    (tmp_path / "skip.py").write_text("b")
    ignored_dir = tmp_path / ".git"
    ignored_dir.mkdir()
    (ignored_dir / "config").write_text("c")

    included = scan_files(tmp_path, ScanOptions(include_globs=("*.md",)))
    assert sorted(entry.path.name for entry in included) == ["keep.md"]

    excluded = scan_files(tmp_path, ScanOptions(exclude_globs=("*.md",)))
    assert [entry.path.name for entry in excluded] == ["skip.py"]

    direct = scan_files(ignored_dir / "config", ScanOptions())
    assert direct == []

    normal_single = scan_files(tmp_path / "keep.md", ScanOptions())
    assert [entry.path.name for entry in normal_single] == ["keep.md"]


def test_classify_file_covers_go_tests_data_and_unknown_roles() -> None:
    assert is_test_path(Path("pkg/server_test.go")) is True
    assert classify_file(Path("data/report.csv")).file_role == FileRole.data
    assert classify_file(Path("blob.weird")).file_role == FileRole.unknown
