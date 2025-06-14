import warnings

warnings.warn(
    "The 'simgrep.models' module is deprecated and will be removed in a future version. " "Import from 'simgrep.core.models' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .core.models import (  # noqa: F401, E402
    Chunk,
    ChunkData,
    OutputMode,
    ProjectConfig,
    SearchResult,
    SimgrepConfig,
)
