import os
from pathlib import Path
from unittest.mock import patch

import pytest

from simgrep.config import SimgrepConfigError, initialize_global_config, load_global_config, save_config
from simgrep.core.models import SimgrepConfig


class TestSimgrepConfig:
    def test_save_config(self, tmp_path: Path) -> None:
        """Test the save_config function."""
        config_file = tmp_path / "test_config.toml"

        with patch("os.path.expanduser", return_value=str(tmp_path)):
            config = SimgrepConfig(config_file=config_file)

        # Modify a value
        config.default_chunk_size_tokens = 512

        # Save it
        save_config(config)

        # Load and verify
        assert config_file.exists()
        reloaded_text = config_file.read_text()
        assert "default_chunk_size_tokens = 512" in reloaded_text

    def test_initialize_and_load_global_config_success(self, tmp_path: Path) -> None:
        """
        Tests successful creation of SimgrepConfig and data directory using mocked home.
        """
        user_home_in_tmp = tmp_path / "userhome"
        user_home_in_tmp.mkdir()  # Ensure the mocked home directory exists

        # Mock os.path.expanduser to control tilde expansion
        def mock_os_expanduser(path_str: str) -> str:
            if path_str == "~" or path_str.startswith("~/"):
                return path_str.replace("~", str(user_home_in_tmp), 1)
            return os.path.expanduser(path_str)  # Fallback for other paths if any

        with patch("os.path.expanduser", side_effect=mock_os_expanduser):
            # 1. Initialize the config
            initialize_global_config()

            # 2. Load the newly created config
            config = load_global_config()

            config_root = user_home_in_tmp / ".config" / "simgrep"
            expected_data_dir = config_root / "default_project"
            expected_config_file = config_root / "config.toml"

            assert isinstance(config, SimgrepConfig)
            assert config.default_project_data_dir == expected_data_dir
            assert config.config_file == expected_config_file
            assert config.default_embedding_model_name == "Qwen/Qwen3-Embedding-0.6B"
            assert config.default_chunk_size_tokens == 128
            assert config.default_chunk_overlap_tokens == 20

            # Check that the directory was created
            assert expected_data_dir.exists()
            assert expected_data_dir.is_dir()
            assert expected_config_file.exists()

            # Check that the default project was created in the global DB
            from simgrep.metadata_db import connect_global_db, get_project_by_name

            global_db_path = config.db_directory / "global_metadata.duckdb"
            conn = connect_global_db(global_db_path)
            try:
                assert get_project_by_name(conn, "default") is not None
            finally:
                conn.close()

    def test_load_global_config_dir_already_exists(self, tmp_path: Path) -> None:
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
            initialize_global_config()
            config = load_global_config()
            assert config.default_project_data_dir == expected_data_dir
            assert config.config_file == expected_config_file
            assert expected_data_dir.exists()
            assert expected_config_file.exists()

    def test_initialize_global_config_permission_error(self, tmp_path: Path) -> None:
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
            # Mock Path.mkdir to simulate a permission issue.
            with patch.object(Path, "mkdir", side_effect=OSError("Permission denied")):
                with pytest.raises(
                    SimgrepConfigError,
                    match="Fatal: Could not create simgrep data directory",
                ):
                    initialize_global_config()

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
        assert config.default_embedding_model_name == "Qwen/Qwen3-Embedding-0.6B"
        assert config.default_chunk_size_tokens == 128
        assert config.default_chunk_overlap_tokens == 20
        # assert config.llm_api_key is None # If/when added
