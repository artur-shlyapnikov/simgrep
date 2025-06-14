import warnings

warnings.warn(
    "The 'simgrep.vector_store' module is deprecated and will be removed in a future version. " "Use 'simgrep.adapters.usearch_index.USearchIndex' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# For backward compatibility, we can try to provide shims.
# This is difficult because the new interface is class-based.
# We will just re-export the new class and supporting types.
from .adapters.usearch_index import USearchIndex  # noqa: F401, E402
from .core.errors import VectorStoreError  # noqa: F401, E402
from .core.models import SearchResult  # noqa: F401, E402
