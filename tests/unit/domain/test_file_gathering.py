import os
from pathlib import Path

import pytest

from simgrep.utils import gather_files_to_process


class TestGatherFilesToProcess:
    def test_single_pattern(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("A")
        (tmp_path / "b.md").write_text("B")
        result = gather_files_to_process(tmp_path, ["*.txt"])
        assert result == [(tmp_path / "a.txt").resolve()]

    def test_multiple_patterns(self, tmp_path: Path) -> None:
        file_txt = tmp_path / "a.txt"
        file_md = tmp_path / "b.md"
        file_csv = tmp_path / "c.csv"
        file_txt.write_text("A")
        file_md.write_text("B")
        file_csv.write_text("C")
        result = gather_files_to_process(tmp_path, ["*.txt", "*.md"])
        assert set(result) == {file_txt.resolve(), file_md.resolve()}

    def test_single_file_path(self, tmp_path: Path) -> None:
        file_path = tmp_path / "only.txt"
        file_path.write_text("content")
        result = gather_files_to_process(file_path, ["*.txt"])
        assert result == [file_path.resolve()]

    def test_empty_directory_no_matches(self, tmp_path: Path) -> None:
        result = gather_files_to_process(tmp_path, ["*.doesnotmatch"])
        assert result == []

    def test_ignored_file_skipped(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("ignored.txt\n")
        (tmp_path / "ignored.txt").write_text("no")
        (tmp_path / "keep.txt").write_text("yes")
        result = gather_files_to_process(tmp_path, ["*.txt"])
        assert result == [(tmp_path / "keep.txt").resolve()]

    def test_explicitly_provided_ignored_file_is_processed(self, tmp_path: Path) -> None:
        """
        Tests that if a user provides a direct path to an ignored file, it is still
        processed, as the explicit user intent overrides the .gitignore.
        """
        (tmp_path / ".gitignore").write_text("skip.txt\n")
        file_path = tmp_path / "skip.txt"
        file_path.write_text("x")
        result = gather_files_to_process(file_path, ["*.txt"])
        # The user's explicit intent to process a single file should override gitignore.
        assert result == [file_path.resolve()]

    def test_gitignore_with_negation_and_dir_patterns(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text(
            """
            # Ignore all logs
            *.log
            # But not this one
            !important.log
            # Ignore build directories
            **/build/
            """
        )
        (tmp_path / "app.log").write_text("log")
        (tmp_path / "important.log").write_text("important")
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        (build_dir / "output.log").write_text("build log")
        result = gather_files_to_process(tmp_path, ["*.log"])
        assert result == [(tmp_path / "important.log").resolve()]

    @pytest.mark.skipif(os.name == "nt", reason="Symlinks work differently on Windows")
    def test_broken_symlink_is_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("A")
        non_existent_target = tmp_path / "non_existent.txt"
        broken_link = tmp_path / "broken.txt"
        os.symlink(non_existent_target, broken_link)

        result = gather_files_to_process(tmp_path, ["*.txt"])
        assert result == [(tmp_path / "a.txt").resolve()]
        # The main thing is that it doesn't raise an exception.
