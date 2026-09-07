# Repository Guidelines

- Prefer simplicity.
- It's a greenfield project: don't worry about breaking changes; you can break interfaces, including public ones.
- RGR TDD by default.
- Performance matters.

## Project Structure & Module Organization

`simgrep/` is the Python package. `simgrep/main.py` exposes the Typer CLI
(`simgrep.main:app`); `mcp_server.py` exposes the stdio MCP transport.
`execution.py` coordinates transport-independent use cases, `corpus.py` manages
searchable corpora, and `search.py`, `indexing.py`, and `store.py` implement the
search/index/persistence pipeline. Concrete embedding, extraction, reranking,
and vector integrations live in `simgrep/adapters/`.

Keep business behavior in application modules and rendering in `output.py` or
transport code. Avoid importing CLI presentation into the storage and adapters.
Tests live in `tests/unit/`, `tests/integration/`, `tests/e2e/`, and
`tests/adapter_external/`. Design notes live in `docs/`.

## Build, Test, and Development Commands

- `just install`: install dev/security dependencies and editable package.
- `just run`: smoke-check CLI wiring with `simgrep --help`.
- `just lint`: Ruff lint (`E`, `F`, `I`).
- `just format` / `just format-check`: apply or verify Ruff format.
- `just test`: fast default suite: unit + integration, excluding `external` and `slow`.
- `just test-unit`, `just test-integration`, `just test-e2e`: focused suites by layer.
- `just test-external`: Hugging Face, `unstructured`, or native-index tests.
- `just test-all`: full suite with coverage.
- `just typecheck`: strict Mypy for `simgrep/` and `tests/`.
- `just security`: pip-audit, Bandit, zizmor, gitleaks.

## Coding Style & Naming Conventions

Explicit types, and small functions. Keep transport parsing and rendering separate from application logic; adapters handle external libraries. Ruff owns imports and formatting (`line-length = 160`).

## Testing Guidelines

Use pytest. Prefer regression-first changes: add or update the failing test, implement the fix, then run the narrowest suite. Unit-test pure logic and service orchestration with fakes from `tests/conftest.py`. Add integration tests for database/index behavior and e2e tests for CLI output. Name files `test_<area>.py`; tests `test_<expected_behavior>`. Mark expensive cases with `external` or `slow`; use `regression` for bug guards. Minimum pre-PR check: `just lint typecheck test`.
