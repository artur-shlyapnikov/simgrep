import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from simgrep.utils import calculate_file_hash, get_project_name_from_local_config


@pytest.fixture
def temp_text_file(tmp_path: Path) -> Path:
    file_content = "Hello World.\nThis is a test file.\nSimgrep is cool."
    file = tmp_path / "test_file.txt"
    file.write_text(file_content, encoding="utf-8")
    return file


class TestCalculateFileHash:
    def test_calculate_file_hash_valid_file(self, temp_text_file: Path) -> None:
        expected = hashlib.sha256(temp_text_file.read_bytes()).hexdigest()
        assert calculate_file_hash(temp_text_file) == expected

    def test_calculate_file_hash_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.txt"
        with pytest.raises(FileNotFoundError):
            calculate_file_hash(missing)

    def test_calculate_file_hash_io_error(self, temp_text_file: Path) -> None:
        """Test IOError handling in calculate_file_hash."""
        with patch("builtins.open", side_effect=OSError("Disk read error")):
            with pytest.raises(IOError, match="Error reading file for hashing"):
                calculate_file_hash(temp_text_file)


class TestGetProjectNameFromLocalConfig:
    def test_get_project_name_success(self, tmp_path: Path) -> None:
        simgrep_dir = tmp_path / ".simgrep"
        simgrep_dir.mkdir()
        (simgrep_dir / "config.toml").write_text('project_name = "my-test-project"')
        assert get_project_name_from_local_config(tmp_path) == "my-test-project"

    def test_get_project_name_file_not_found(self, tmp_path: Path) -> None:
        simgrep_dir = tmp_path / ".simgrep"
        simgrep_dir.mkdir()
        # No config.toml
        assert get_project_name_from_local_config(tmp_path) is None

    def test_get_project_name_malformed_toml(self, tmp_path: Path) -> None:
        simgrep_dir = tmp_path / ".simgrep"
        simgrep_dir.mkdir()
        (simgrep_dir / "config.toml").write_text('project_name = "my-test-project')  # Missing quote
        assert get_project_name_from_local_config(tmp_path) is None

    def test_get_project_name_empty_toml(self, tmp_path: Path) -> None:
        simgrep_dir = tmp_path / ".simgrep"
        simgrep_dir.mkdir()
        (simgrep_dir / "config.toml").write_text("")
        assert get_project_name_from_local_config(tmp_path) is None

    def test_get_project_name_key_missing(self, tmp_path: Path) -> None:
        simgrep_dir = tmp_path / ".simgrep"
        simgrep_dir.mkdir()
        (simgrep_dir / "config.toml").write_text('other_key = "some_value"')
        assert get_project_name_from_local_config(tmp_path) is None
