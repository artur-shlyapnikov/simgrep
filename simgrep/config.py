import sys  # for printing to stderr

try:
    from .models import SimgrepConfig
except ImportError:
    from simgrep.models import SimgrepConfig


# default number of top results to fetch for searches
DEFAULT_K_RESULTS = 5


class SimgrepConfigError(Exception):
    """Custom exception for simgrep configuration errors."""

    pass


def load_or_create_global_config() -> SimgrepConfig:
    """
    Instantiates a SimgrepConfig object with default values and ensures
    the default project's data directory exists.

    for deliverable 3.1, no toml file reading/writing is performed.

    Returns:
        An instance of SimgrepConfig.

    Raises:
        SimgrepConfigError: If the data directory cannot be created.
    """
    config = SimgrepConfig()

    # ensure the data directory for the default project exists
    try:
        # config.default_project_data_dir is, e.g., ~/.config/simgrep/default_project
        # mkdir(parents=true) will create ~/.config/simgrep/ as well if it doesn't exist.
        config.default_project_data_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        error_message = (
            f"Fatal: Could not create simgrep data directory at "
            f"'{config.default_project_data_dir}'. Please check permissions. Error: {e}"
        )
        # for now, print to stderr and raise a custom error.
        # in a more mature cli, rich console might be used here.
        print(error_message, file=sys.stderr)
        raise SimgrepConfigError(error_message) from e

    return config