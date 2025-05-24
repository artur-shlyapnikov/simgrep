# simgrep/exceptions.py
class SimgrepDBError(Exception):
    """Base exception for simgrep database/store errors."""
    pass

class MetadataDBError(SimgrepDBError):
    """Custom exception for DuckDB specific errors."""
    pass

class VectorStoreError(SimgrepDBError):
    """Custom exception for USearch specific errors."""
    pass

# IndexerError is defined in indexer.py to avoid circular imports if indexer needs SimgrepDBError.
# However, if it were to be a general SimgrepError, it could be here.
# For now, keeping it in indexer.py as per plan.
# class IndexerError(SimgrepError):
#     """Custom exception for errors during the indexing process."""
#     pass