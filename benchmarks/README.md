# Speed Benchmark Suite for simgrep

This directory contains the speed-benchmark harness for measuring user-visible CLI latency and white-box service phase timings.

## Suites

### CI Suite (`--suite ci`)

Purpose: Stable regression detection in PRs.

Characteristics:
- No network
- No model downloads
- No GPU dependency
- Uses deterministic generated corpora
- Uses fixed worker counts (usually `workers=1`)
- Uses fake deterministic embedder through in-process service benchmarks

Cases:
- `scan.medium` - File discovery + ignore handling
- `index.rebuild.medium.fake` - Full dense indexing without model noise
- `index.incremental.noop.medium.fake` - Unchanged-index overhead
- `search.persistent.skip.medium.fake` - Warm persistent search without freshness

### Real-Model Suite (`--suite real`)

Purpose: Local or nightly measurement of actual user-perceived speed.

Characteristics:
- Uses actual default model: `ibm-granite/granite-embedding-30m-english`
- Requires model/tokenizer cached
- Skips with clear reason if cache is missing
- Runs CLI subprocesses, not only service calls
- Uses isolated temporary HOME

Cases:
- `cli.index.rebuild.small.real` - Real first rebuild excluding download
- `cli.search.persistent.skip.small.real` - Real process startup + load + query

### Stress Suite (`--suite stress`)

Purpose: Manual scalability profiling (not yet implemented).

Characteristics:
- Large generated corpus
- Not run in normal CI
- Measures throughput and asymptotic behavior

## Running Benchmarks

### Quick Start

```bash
# Run CI suite
uv run python benchmarks/run_speed.py --suite ci --output reports/benchmarks/speed-ci.json

# Run real-model suite (requires cached model)
uv run python benchmarks/run_speed.py --suite real --output reports/benchmarks/speed-real.json

# Compare to baseline
uv run python benchmarks/run_speed.py --suite ci --compare benchmarks/baselines/ci.json

# Update baseline with current results
uv run python benchmarks/run_speed.py --suite ci --update-baseline --compare benchmarks/baselines/ci.json
```

### just Recipes

```bash
just bench-speed        # Run CI suite with comparison
just bench-speed-record # Run CI suite, record results
just bench-speed-real   # Run real-model suite
```

## Corpus Profiles

| Profile | Files | Approx Size | Purpose |
|---------|------:|------------:|---------|
| `tiny` | 20 | < 200 KB | Real CLI smoke |
| `small` | 100 | 1–2 MB | Real-model local/nightly |
| `medium` | 1,000 | 8–15 MB | CI fake regression |
| `large` | 10,000 | 80–150 MB | Manual stress |

## Baseline Management

Baselines are stored in `benchmarks/baselines/`.

### Calibration Process

1. **Initial recording**: Run in record-only mode for several CI runs
   ```bash
   uv run python benchmarks/run_speed.py --suite ci --output reports/benchmarks/speed-ci.json
   ```

2. **Baseline creation**: After stabilization, copy to baseline
   ```bash
   cp reports/benchmarks/speed-ci.json benchmarks/baselines/ci.json
   ```

3. **Enforcement**: Enable comparison in CI
   ```bash
   uv run python benchmarks/run_speed.py --suite ci --compare benchmarks/baselines/ci.json
   ```

### Baseline Format

```json
{
  "schema_version": 1,
  "cases": {
    "index.rebuild.medium.fake": {
      "median_seconds": 1.23,
      "p95_seconds": 1.40,
      "max_regression_ratio": 1.25,
      "hard_max_seconds": 3.00
    }
  }
}
```

### Comparison Rules

- Primary gate: Median must not exceed `baseline_median * max_regression_ratio`
- Secondary gate: P95 must not exceed `baseline_p95 * max_regression_ratio`
- Hard ceiling: Optional, catches extreme regressions

## Report Format

Reports are written as JSON with this structure:

```json
{
  "schema_version": 1,
  "created_at": "2024-01-01T00:00:00Z",
  "git_commit": "abc123",
  "environment": {
    "python_version": "3.12.0",
    "platform": "macOS-14.0-arm64-arm-64bit",
    "machine": "arm64",
    "cpu_count": 10,
    "os": "Darwin"
  },
  "simgrep": {
    "version": "0.1.0",
    "default_model": "ibm-granite/granite-embedding-30m-english",
  },
  "benchmarks": [
    {
      "name": "index.rebuild.medium.fake",
      "suite": "ci",
      "corpus": {
        "profile": "medium",
        "files": 1000,
        "bytes": 15000000,
        "expected_indexable_files": 800
      },
      "config": {
        "workers": 1,
        "batch_size": 64,
        "model_mode": "fake"
      },
      "iterations": 10,
      "warmups": 2,
      "samples_seconds": [1.2, 1.3, 1.25, ...],
      "median_seconds": 1.28,
      "p95_seconds": 1.45,
      "min_seconds": 1.20,
      "max_seconds": 1.50,
      "phases": {
        "plan_seconds": 0.05,
        "scan_seconds": 0.10,
        "extract_chunk_seconds": 0.30,
        "embedding_seconds": 0.50,
        "store_seconds": 0.20,
        "index_save_seconds": 0.10
      },
      "outcome": {
        "files_indexed": 800,
        "chunks_indexed": 5000,
        "errors": 0
      },
      "status": {
        "passed": true,
        "skipped": false
      }
    }
  ]
}
```

## Known Issues

### Freshness Auto-Refresh

The `search.auto.one_changed.medium.fake` benchmark is not yet implemented because:

> Current code likely violates this because freshness auto sets `wipe=True` for any mutation.

The benchmark should start report-only until the freshness behavior is fixed. This is a known performance regression risk.

## Implementation Notes

### Phase Timing Attribution

| Phase | Attribution |
|-------|-------------|
| `plan_seconds` | `Indexer.run_index()` - file plan building |
| `scan_seconds` | `IndexService` - file discovery |
| `extract_chunk_seconds` | `IndexService` - extraction + chunking |
| `embedding_seconds` | `IndexService` - model encoding |
| `store_seconds` | `IndexService` - DB + vector index writes |
| `index_save_seconds` | `Indexer` - final index serialization |
| `index_load_seconds` | Backend - USearch index loading |
| `query_embedding_seconds` | `SearchService` - query encoding |
| `vector_search_seconds` | `SearchService` - ANN search |
| `ranking_seconds` | `CandidateRanker` - reranking |
| `freshness_seconds` | `SearchRunner` - freshness check |
