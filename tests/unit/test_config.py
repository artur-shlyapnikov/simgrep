import os
from pathlib import Path
from unittest.mock import patch

import pytest

from simgrep.config import SimgrepConfigError, load_or_create_global_config
from simgrep.models import SimgrepConfig


class TestSimgrepConfig:
    def test_load_or_create_global_config_success(self, tmp_path: Path) -> None:
        """
        Tests successful creation of SimgrepConfig and data directory using mocked home.
        """
        user_home_in_tmp = tmp_path / "userhome"
        user_home_in_tmp.mkdir()  # Ensure the mocked home directory exists

        config_root = user_home_in_tmp / ".config" / "simgrep"
        expected_data_dir = config_root / "default_project"
        expected_config_file = config_root / "config.toml"

        # Mock os.path.expanduser to control tilde expansion
        def mock_os_expanduser(path_str: str) -> str:
            if path_str == "~" or path_str.startswith("~/"):
                return path_str.replace("~", str(user_home_in_tmp), 1)
            return os.path.expanduser(path_str)  # Fallback for other paths if any

        with patch("os.path.expanduser", side_effect=mock_os_expanduser):
            config = load_or_create_global_config()

            assert isinstance(config, SimgrepConfig)
            assert config.default_project_data_dir == expected_data_dir
            assert config.config_file == expected_config_file
            assert config.default_embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
            assert config.default_chunk_size_tokens == 128
            assert config.default_chunk_overlap_tokens == 20

            # Check that the directory was created
            assert expected_data_dir.exists()
            assert expected_data_dir.is_dir()
            assert expected_config_file.exists()
            assert "default" in config.projects

    def test_load_or_create_global_config_dir_already_exists(self, tmp_path: Path) -> None:
        """
        Tests that function works if the directory already exists.
        """
        user_home_in_tmp = tmp_path / "userhome"
        user_home_in_tmp.mkdir()
        config_root = user_home_in_tmp / ".config" / "simgrep"
        expected_data_dir = config_root / "default_project"
        expected_config_file = config_root / "config.toml"
        expected_data_dir.mkdir(parents=True, exist_ok=True)

        def mock_os_expanduser(path_str: str) -> str:
            if path_str == "~" or path_str.startswith("~/"):
                return path_str.replace("~", str(user_home_in_tmp), 1)
            return os.path.expanduser(path_str)

        with patch("os.path.expanduser", side_effect=mock_os_expanduser):
            config = load_or_create_global_config()
            assert config.default_project_data_dir == expected_data_dir
            assert config.config_file == expected_config_file
            assert expected_data_dir.exists()
            assert expected_config_file.exists()

    def test_load_or_create_global_config_permission_error(self, tmp_path: Path) -> None:
        """
        Tests that SimgrepConfigError is raised if directory creation fails.
        """
        user_home_in_tmp = tmp_path / "userhome"
        # We don't create user_home_in_tmp.mkdir() here to ensure it's part of the test.
        # The critical part is mocking Path.mkdir to raise an error.

        def mock_os_expanduser(path_str: str) -> str:
            if path_str == "~" or path_str.startswith("~/"):
                return path_str.replace("~", str(user_home_in_tmp), 1)
            return os.path.expanduser(path_str)

        # Patch os.path.expanduser to control the path resolution
        with patch("os.path.expanduser", side_effect=mock_os_expanduser):
            # Mock the SimgrepConfig instance that will be created to ensure its
            # default_project_data_dir is what we expect, then mock mkdir on that.
            # This is a bit complex. A simpler way is to trust SimgrepConfig resolves the path
            # correctly under the os.path.expanduser patch, and then directly patch Path.mkdir.

            # The SimgrepConfig will instantiate with default_project_data_dir = resolved_path_under_tmp
            # due to the mock_os_expanduser.
            # So, we need to patch Path.mkdir globally for this test.
            with patch.object(Path, "mkdir", side_effect=OSError("Permission denied")):
                with pytest.raises(
                    SimgrepConfigError,
                    match="Fatal: Could not create simgrep data directory",
                ):
                    load_or_create_global_config()

    def test_simgrep_config_defaults(self) -> None:
        """Test that SimgrepConfig model has correct default values."""
        # This test should ideally not rely on the actual user's home.
        # We can mock expanduser here too for consistency if needed,
        # but for just checking Pydantic defaults, direct inspection is fine.
        # However, to make it fully independent:

        mock_home = Path("/mock/home")

        def mock_os_expanduser_for_defaults(path_str: str) -> str:
            if path_str == "~" or path_str.startswith("~/"):
                return path_str.replace("~", str(mock_home), 1)
            return os.path.expanduser(path_str)  # Fallback

        with patch("os.path.expanduser", side_effect=mock_os_expanduser_for_defaults):
            config = SimgrepConfig()
            expected_default_dir = mock_home / ".config" / "simgrep" / "default_project"
            assert config.default_project_data_dir == expected_default_dir
            assert config.config_file == mock_home / ".config" / "simgrep" / "config.toml"
            assert config.db_directory == mock_home / ".config" / "simgrep"

        # These defaults don't depend on path expansion
        assert config.default_embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.default_chunk_size_tokens == 128
        assert config.default_chunk_overlap_tokens == 20
        # assert config.llm_api_key is None # If/when added