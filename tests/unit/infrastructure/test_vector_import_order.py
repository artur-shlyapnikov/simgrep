"""Import-order regression tests for simgrep.adapters.vector.

usearch and torch each bundle an OpenMP runtime; whichever library loads
first owns the process, and torch parallel kernels segfault in libomp
barriers when usearch's runtime loaded first. A segfault cannot be asserted
in-process, so these tests spawn subprocesses.

Invariant since the lazy-runtime work: simgrep processes are safe because
they either (a) never import torch (ONNX query path), (b) import torch
before the first USearchIndex (eager fallbacks / require_bulk), or
(c) mark torch pending so the usearch guard imports torch first.
"""

from __future__ import annotations

import subprocess
import sys

LIGHT_SCRIPT = (
    "import sys\n"
    "import simgrep.adapters.vector\n"
    "assert 'usearch.index' not in sys.modules, 'usearch.index imported at module load'\n"
    "assert 'torch' not in sys.modules, 'torch imported at module load'\n"
    "print('light')\n"
)
USEARCH_THEN_TORCH_SCRIPT = (
    "import simgrep.adapters.vector as v\n"
    "import numpy as np\n"
    "idx = v.USearchIndex(ndim=4)\n"
    "idx.add(labels=np.arange(3, dtype=np.int64), vectors=np.eye(3, 4, dtype=np.float32))\n"
    "import torch\n"
    "torch.randn(64, 4).clone().to(torch.float16)\n"
    "a = torch.randn(2048, 2048)\n"
    "b = torch.randn(2048, 2048)\n"
    "total = float((a @ b).sum())\n"
    "hits = idx.search(np.eye(1, 4, dtype=np.float32), 2)\n"
    "print('no-crash', len(hits), total)\n"
)


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_importing_vector_module_stays_light() -> None:
    result = _run(LIGHT_SCRIPT)
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "light" in result.stdout


def test_usearch_only_process_never_imports_torch() -> None:
    """Query-only flows: usearch loads, torch never arrives — safe by absence."""
    script = (
        "import sys\n"
        "import simgrep.adapters.vector as v\n"
        "import numpy as np\n"
        "idx = v.USearchIndex(ndim=4)\n"
        "idx.add(labels=np.arange(3, dtype=np.int64), vectors=np.eye(3, 4, dtype=np.float32))\n"
        "hits = idx.search(np.eye(1, 4, dtype=np.float32), 2)\n"
        "assert len(hits) == 2\n"
        "assert 'torch' not in sys.modules, 'guard imported torch without a pending marker'\n"
        "print('usearch-only')\n"
    )
    result = _run(script)
    assert result.returncode == 0, f"returncode={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "usearch-only" in result.stdout


def test_pending_torch_guard_loads_torch_before_usearch() -> None:
    """Bulk flows: a pending lazy embedder makes the guard import torch first,
    so later torch parallel kernels run under torch's own libomp."""
    script = USEARCH_THEN_TORCH_SCRIPT.replace(
        "import simgrep.adapters.vector as v\n",
        "import sys\nimport simgrep.adapters.vector as v\n",
    ).replace(
        "import numpy as np\n",
        "import numpy as np\nv.mark_torch_pending()\n",
    )
    result = _run(script)
    assert result.returncode == 0, f"returncode={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "no-crash" in result.stdout
