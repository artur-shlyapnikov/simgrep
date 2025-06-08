from pathlib import Path

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

    def test_path_is_file(self, tmp_path: Path) -> None:
        file_path = tmp_path / "only.txt"
        file_path.write_text("X")
        result = gather_files_to_process(file_path, ["*.txt"])
        assert result == [file_path.resolve()]

    def test_nested_directories(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub" / "nested"
        sub.mkdir(parents=True)
        f1 = sub / "a.txt"
        f2 = sub / "b.md"
        f1.write_text("1")
        f2.write_text("2")
        result = gather_files_to_process(tmp_path, ["*.txt", "*.md"])
        assert set(result) == {f1.resolve(), f2.resolve()}

    def test_no_matching_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("A")
        result = gather_files_to_process(tmp_path, ["*.md"])
        assert result == []

    def test_deduplicate_and_sort(self, tmp_path: Path) -> None:
        f1 = tmp_path / "c.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("1")
        f2.write_text("2")
        result = gather_files_to_process(tmp_path, ["*.txt", "c.*", "*.txt"])
        assert result == [f2.resolve(), f1.resolve()]
