class SimgrepDBError(Exception):
    """Base exception for simgrep database/store errors."""

    pass


class MetadataDBError(SimgrepDBError):
    """Custom exception for DuckDB specific errors."""

    pass


class VectorStoreError(SimgrepDBError):
    """Custom exception for USearch specific errors."""

    pass


class IndexerError(SimgrepDBError):
    """Custom exception for errors during the indexing process."""

    pass
