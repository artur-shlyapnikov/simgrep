from pathlib import Path

from rich.console import Console

import pytest

from simgrep.formatter import format_paths


class TestFormatPaths:
    def test_unique_sorted_absolute_paths(self, tmp_path: Path) -> None:
        file_a = tmp_path / "a.txt"
        file_b = tmp_path / "b.txt"
        file_a.write_text("a")
        file_b.write_text("b")

        paths = [file_b, file_a, file_b]
        result = format_paths(paths, use_relative=False, base_path=None)

        expected = "\n".join([str(file_a.resolve()), str(file_b.resolve())])
        assert result == expected

    def test_relative_paths_with_base_path(self, tmp_path: Path) -> None:
        base_path = tmp_path / "base"
        dir1 = base_path / "dir1"
        dir2 = base_path / "dir2"
        dir1.mkdir(parents=True)
        dir2.mkdir(parents=True)
        file1 = dir1 / "file1.txt"
        file2 = dir2 / "file2.txt"
        file1.write_text("1")
        file2.write_text("2")

        result = format_paths([file2, file1], use_relative=True, base_path=base_path)
        expected = "\n".join(
            sorted(
                [
                    str(file1.relative_to(base_path)),
                    str(file2.relative_to(base_path)),
                ]
            )
        )
        assert result == expected

    def test_relative_paths_missing_base_warns_and_falls_back_to_absolute(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        file1 = tmp_path / "f1.txt"
        file2 = tmp_path / "f2.txt"
        file1.write_text("1")
        file2.write_text("2")

        test_console = Console()
        result = format_paths([file1, file2], use_relative=True, base_path=None, console=test_console)
        captured = capsys.readouterr()
        assert "base_path was not provided to format_paths" in captured.out

        expected = "\n".join(
            sorted(
                [
                    str(file1.resolve()),
                    str(file2.resolve()),
                ]
            )
        )
        assert result == expected
