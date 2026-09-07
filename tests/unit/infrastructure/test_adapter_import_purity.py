"""Contract: importing adapter modules must stay cheap -- heavy ML deps load lazily."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

PROBE = """
import sys

import simgrep.adapters.chunker
import simgrep.adapters.embedder

heavy = {"transformers", "sentence_transformers", "torch", "huggingface_hub"}
loaded = sorted(set(sys.modules) & heavy)
assert not loaded, f"adapter import pulled in heavy deps: {loaded}"
"""


def test_importing_adapter_modules_does_not_load_heavy_dependencies() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=60)

    assert result.returncode == 0, result.stderr


CRASH_PROBE = """
import simgrep.main  # noqa: F401  (full CLI module set must load first)

from simgrep.adapters.embedder import SentenceEmbedder

embedder = SentenceEmbedder(model_name="ibm-granite/granite-embedding-30m-english", normalize_embeddings=True)
vectors = embedder.encode(["needle"])
assert vectors.shape[0] == 1
"""


@pytest.mark.slow
@pytest.mark.regression
def test_full_cli_import_set_then_embedder_encode_does_not_crash() -> None:
    """Regression guard (round 11): cold CLI process that imports simgrep.main first and
    only then lazily loads transformers/SentenceTransformer used to SIGSEGV deterministically."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", CRASH_PROBE],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    assert result.returncode == 0, f"child exited {result.returncode}: {result.stderr[-500:]}"
