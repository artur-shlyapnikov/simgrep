from pathlib import Path

import pytest
from rich.console import Console

from simgrep.formatter import format_count, format_paths, format_show_basic
from simgrep.models import ChunkData


class TestFormatShowBasic:
    def test_basic_formatting(self, tmp_path: Path) -> None:
        file_path = tmp_path / "file.txt"
        result = format_show_basic(file_path, "snippet", 0.123456)
        expected = f"File: {str(file_path)}\nScore: 0.1235\nChunk: snippet"
        assert result == expected


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
    ) -> None:
        file1 = tmp_path / "f1.txt"
        file2 = tmp_path / "f2.txt"
        file1.write_text("1")
        file2.write_text("2")

        test_console = Console(record=True)
        result = format_paths(
            [file1, file2],
            use_relative=True,
            base_path=None,
            console=test_console,
        )
        output_text = test_console.export_text()
        assert "base_path was not provided to format_paths" in output_text

        expected = "\n".join(
            sorted(
                [
                    str(file1.resolve()),
                    str(file2.resolve()),
                ]
            )
        )
        assert result == expected

    def test_relative_paths_outside_base_fallbacks_to_absolute(self, tmp_path: Path) -> None:
        base_path = tmp_path / "base"
        base_path.mkdir()
        outside_file = tmp_path / "outside.txt"
        outside_file.write_text("x")

        result = format_paths([outside_file], use_relative=True, base_path=base_path)

        assert result == str(outside_file.resolve())


class TestFormatCount:
    def test_counts_unique_files_and_chunks(self) -> None:
        results = [
            ChunkData(
                text="a",
                source_file_path=Path("/tmp/a.txt"),
                source_file_id=0,
                usearch_label=0,
                start_char_offset=0,
                end_char_offset=1,
                token_count=1,
            ),
            ChunkData(
                text="b",
                source_file_path=Path("/tmp/a.txt"),
                source_file_id=0,
                usearch_label=1,
                start_char_offset=2,
                end_char_offset=3,
                token_count=1,
            ),
            ChunkData(
                text="c",
                source_file_path=Path("/tmp/b.txt"),
                source_file_id=1,
                usearch_label=2,
                start_char_offset=0,
                end_char_offset=1,
                token_count=1,
            ),
        ]

        result = format_count(results)
        assert result == "3 matching chunks in 2 files."
