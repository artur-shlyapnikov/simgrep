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
