"""Shared fixtures for unit tests."""

# Cold first-import of torch/sentence_transformers under 9-way xdist contention exceeds
# the per-test --timeout window; importing at collection time (outside pytest-timeout)
# also pins the safe torch-before-usearch order in every xdist worker.
try:
    import sentence_transformers  # noqa: F401
except ImportError:
    pass
import torch  # noqa: F401  # must precede any usearch import
