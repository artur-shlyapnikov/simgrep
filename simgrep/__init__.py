"""Single-threaded OpenMP for the whole process.

torch's libomp worker threads race at interpreter teardown after duckdb and
usearch activity in the same process: short encode flows and index deletions
segfault inside ``__kmp_hyper_barrier_release``. Encode throughput is GPU-bound
(MPS), so a single OpenMP thread costs nothing measurable (473 vs 472 ch/s
interleaved A/B) while removing the entire crash class. Must be set before any
native extension loads, hence at package import time.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
