from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from simgrep.config import load_or_create_global_config, SimgrepConfigError
from simgrep.models import SimgrepConfig


class TestSimgrepConfig:
    def test_load_or_create_global_config_success(self, tmp_path: Path):
        """
        Tests successful creation of SimgrepConfig and data directory.
        """
        # Mock expanduser to redirect the default_project_data_dir to tmp_path
        # SimgrepConfig.default_project_data_dir uses default_factory,
        # so we patch Path.expanduser which is called by it.
        mock_expanduser_path = tmp_path / ".config" / "simgrep" / "default_project"

        with patch("pathlib.Path.expanduser", return_value=mock_expanduser_path.parent.parent.parent):
            # Adjust the return value to match how expanduser is typically used for ~/.config
            # The factory is lambda: Path("~/.config/simgrep/default_project").expanduser()
            # So, if expanduser() returns /tmp/somepath/.config/simgrep/default_project
            # then Path("~") becomes /tmp/somepath
            # and Path("~/.config/simgrep/default_project") becomes /tmp/somepath/.config/simgrep/default_project

            # To make it simpler, let's mock the SimgrepConfig model's field directly for this test
            # if direct mocking of default_project_data_dir is cleaner.
            # The current SimgrepConfig has default_project_data_dir as a Field with default_factory.
            # Let's patch the factory's output or the path creation within the function.

            # The path created is config.default_project_data_dir
            # config = SimgrepConfig() -> this sets default_project_data_dir
            # We need this path to be under tmp_path.

            # Let's patch SimgrepConfig instantiation to control its default_project_data_dir
            # or patch the mkdir call.

            # Simpler: Patch `Path.mkdir` to see if it's called correctly, and control `default_project_data_dir`
            # via `SimgrepConfig` if possible, or ensure the hardcoded path logic is tested but redirected.

            # The `default_project_data_dir` is resolved within SimgrepConfig's default_factory.
            # We need `Path("~/.config/simgrep/default_project").expanduser()` to point to tmp_path.
            
            # Path("~").expanduser() is the key.
            # If Path("~").expanduser() returns `tmp_path / "userhome"`, then
            # Path("~/.config/simgrep/default_project").expanduser() would be
            # `tmp_path / "userhome" / ".config" / "simgrep" / "default_project"`
            
            user_home_in_tmp = tmp_path / "userhome"
            user_home_in_tmp.mkdir()
            
            expected_data_dir = user_home_in_tmp / ".config" / "simgrep" / "default_project"

            with patch("pathlib.Path.home", return_value=user_home_in_tmp):
                config = load_or_create_global_config()

                assert isinstance(config, SimgrepConfig)
                assert config.default_project_data_dir == expected_data_dir
                assert config.default_embedding_model_name == "all-MiniLM-L6-v2"
                assert config.default_chunk_size_tokens == 128
                assert config.default_chunk_overlap_tokens == 20

                # Check that the directory was created
                assert expected_data_dir.exists()
                assert expected_data_dir.is_dir()

    def test_load_or_create_global_config_dir_already_exists(self, tmp_path: Path):
        """
        Tests that function works if the directory already exists.
        """
        user_home_in_tmp = tmp_path / "userhome"
        user_home_in_tmp.mkdir()
        expected_data_dir = user_home_in_tmp / ".config" / "simgrep" / "default_project"
        expected_data_dir.mkdir(parents=True, exist_ok=True) # Pre-create the directory

        with patch("pathlib.Path.home", return_value=user_home_in_tmp):
            config = load_or_create_global_config()
            assert config.default_project_data_dir == expected_data_dir
            assert expected_data_dir.exists() # Still exists

    def test_load_or_create_global_config_permission_error(self, tmp_path: Path):
        """
        Tests that SimgrepConfigError is raised if directory creation fails.
        """
        user_home_in_tmp = tmp_path / "userhome"
        # Not creating user_home_in_tmp to make .mkdir fail if it tries to create parents
        # A more direct way is to mock mkdir to raise OSError.

        # We want config.default_project_data_dir.mkdir() to fail.
        # Let's assume SimgrepConfig() correctly resolves its path.
        # Then we patch the mkdir method on that specific Path object.

        # Path.home() is used by .expanduser().
        # SimgrepConfig() -> self.default_project_data_dir = Path("~/.config/simgrep/default_project").expanduser()
        # load_or_create_global_config() -> config.default_project_data_dir.mkdir()

        resolved_path_under_tmp = tmp_path / "userhome" / ".config" / "simgrep" / "default_project"

        with patch("pathlib.Path.home", return_value=tmp_path / "userhome"):
            # Mock the SimgrepConfig instance that will be created
            mock_config_instance = SimgrepConfig()
            # Override its default_project_data_dir to our controlled path for the purpose of mocking mkdir
            mock_config_instance.default_project_data_dir = resolved_path_under_tmp
            
            # Patch the SimgrepConfig constructor to return our mock instance
            with patch("simgrep.config.SimgrepConfig", return_value=mock_config_instance):
                # Now, mock the mkdir method on the Path object that will be used
                with patch.object(Path, "mkdir", side_effect=OSError("Permission denied")):
                    with pytest.raises(SimgrepConfigError, match="Fatal: Could not create simgrep data directory"):
                        load_or_create_global_config()
                    
                    # Ensure mkdir was called on the correct path
                    # This is tricky because Path.mkdir is a method of Path class,
                    # and we need to check it was called on `resolved_path_under_tmp`.
                    # A simpler check is that the error is raised.
                    # To verify the path, we'd need a more complex mock setup for Path instances.

    def test_simgrep_config_defaults(self):
        """Test that SimgrepConfig model has correct default values."""
        config = SimgrepConfig()
        
        # Check default factory for default_project_data_dir
        # This will use the actual user's home, which is fine for just checking the path structure.
        expected_default_dir = Path("~/.config/simgrep/default_project").expanduser()
        assert config.default_project_data_dir == expected_default_dir
        
        assert config.default_embedding_model_name == "all-MiniLM-L6-v2"
        assert config.default_chunk_size_tokens == 128
        assert config.default_chunk_overlap_tokens == 20
        # assert config.llm_api_key is None # If/when added
        # assert config.projects == []      # If/when added
    