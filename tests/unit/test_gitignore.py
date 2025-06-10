from pathlib import Path

from simgrep.utils import gather_files_to_process


class TestGitIgnoreSupport:
    def test_ignored_file_skipped(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("ignored.txt\n")
        (tmp_path / "ignored.txt").write_text("no")
        (tmp_path / "keep.txt").write_text("yes")
        result = gather_files_to_process(tmp_path, ["*.txt"])
        assert result == [(tmp_path / "keep.txt").resolve()]

    def test_single_file_path_respects_gitignore(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("skip.txt\n")
        file_path = tmp_path / "skip.txt"
        file_path.write_text("x")
        result = gather_files_to_process(file_path, ["*.txt"])
        assert result == []
