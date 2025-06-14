import warnings

warnings.warn(
    "The 'simgrep.exceptions' module is deprecated and will be removed in a future version. " "Import from 'simgrep.core.errors' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .core.errors import (  # noqa: F401, E402
    IndexerError,
    MetadataDBError,
    SimgrepDBError,
    SimgrepError,
    VectorStoreError,
)
