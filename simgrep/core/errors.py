# simgrep/core/errors.py


class SimgrepError(Exception):
    """Base exception for simgrep."""

    pass


class SimgrepConfigError(SimgrepError):
    """Configuration-related errors."""

    pass


class VectorStoreError(SimgrepError):
    """Vector store related errors."""

    pass


class MetadataDBError(SimgrepError):
    """Metadata database related errors."""

    pass


class IndexerError(SimgrepError):
    """Errors during indexing."""

    pass
