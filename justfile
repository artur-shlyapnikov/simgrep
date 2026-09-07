uv := "uv"
pytest := uv + " run pytest -n auto --cov=simgrep"
reports := "reports"
bench_reports := "reports/benchmarks"

default:
    @just --list

# Install dev/security dependencies and the editable package
[group("setup")]
install:
    {{uv}} sync --locked --group dev --group security

# Check .python-version exists and uv sees the environment
[group("setup")]
check-python-version:
    @if [ ! -f .python-version ]; then echo "Error: .python-version file not found. Please create it with your Python version (e.g., 3.12)."; exit 1; fi
    {{uv}} version

# Pre-download and cache Hugging Face models used by the project
[group("setup")]
download-models:
    {{uv}} run python scripts/cache_hf_model.py

# Full bootstrap: check python, install, cache models
[group("setup")]
setup: check-python-version install download-models
    @echo "Setup complete."

# Run Ruff check
[group("check")]
lint:
    {{uv}} run ruff check .

# Apply Ruff formatting
[group("check")]
format:
    {{uv}} run ruff format .

# Verify Ruff formatting
[group("check")]
format-check:
    {{uv}} run ruff format --check .

# Run Mypy over package and tests
[group("check")]
typecheck:
    {{uv}} run mypy simgrep/ tests/

# Audit installed dependencies against published advisories
[group("check")]
audit:
    {{uv}} run pip-audit

# Scan GitHub Actions workflows with zizmor
[group("check")]
workflow-lint:
    {{uv}} run zizmor .github/workflows

# Scan the repo for leaked secrets (needs gitleaks on PATH)
[group("check")]
secrets-scan:
    @if command -v gitleaks >/dev/null 2>&1; then gitleaks git --redact; else echo "gitleaks is not installed; install from https://github.com/gitleaks/gitleaks/releases"; exit 1; fi

# Full security gate: audit, bandit (medium+), workflow lint, secret scan
[group("check")]
security: audit workflow-lint secrets-scan
    # -ll: gate on medium+ severity. Audited LOW set: B404/B603/B607 subprocess
    # sites use list args with fixed commands (git, sys.executable), no shell=True;
    # Low-severity findings remain visible in the report.
    {{uv}} run bandit -r simgrep -c pyproject.toml -ll

# Fast default suite: unit + integration, excluding external/slow
[group("test")]
test:
    @just _pytest "tests/unit tests/integration -m 'not external and not slow' --timeout=30" junit.xml

# Unit tests (non-external)
[group("test")]
test-unit:
    @just _pytest "tests/unit/ -m 'not external' --timeout=10" junit-unit.xml

# Property tests
[group("test")]
test-property:
    @just _pytest "tests/adapter_external/test_hf_chunker_property.py --timeout=20" junit-property.xml

# Integration tests (non-external)
[group("test")]
test-integration:
    @just _pytest "tests/integration/ -m 'not external' --timeout=20" junit-integration.xml

# End-to-end CLI tests (non-external)
[group("test")]
test-e2e:
    @just _pytest "tests/e2e/ -m 'not external' --timeout=30" junit-e2e.xml

# External adapter/model tests (needs network + cached models)
[group("test")]
test-external:
    @just _pytest "-m external --timeout=60" junit-external.xml

# Slow tests
[group("test")]
test-slow:
    @just _pytest "-m slow --timeout=60" junit-slow.xml

# Full suite with coverage
[group("test")]
test-all:
    @just _pytest "tests/ --timeout=60" junit-all.xml

# CI speed suite with baseline comparison
[group("bench")]
bench-speed:
    @just _bench ci speed-ci.json "--compare benchmarks/baselines/ci.json"

# CI speed suite, record results without comparison
[group("bench")]
bench-speed-record:
    @just _bench ci speed-ci.json ""

# Real-model speed suite
[group("bench")]
bench-speed-real:
    @just _bench real speed-real.json ""

# Stress speed suite (manual)
[group("bench")]
bench-speed-stress:
    @just _bench stress speed-stress.json ""

# Run the simgrep CLI (default: --help); pass through args, e.g. `just run search ".." .`
[positional-arguments]
run *args="--help":
    {{uv}} run simgrep "$@"

# Remove build artifacts and caches (incl. .venv)
clean:
    find . -type f -name '*.py[co]' -delete
    find . -type d -name '__pycache__' -exec rm -rf {} +
    rm -rf .pytest_cache .mypy_cache build/ dist/ *.egg-info/ .venv

# Local verification without model downloads or security services
check: lint format-check typecheck test test-e2e

# All checks: install, lint, format-check, test, typecheck, security
all: install lint format-check test typecheck security
    @echo "All checks passed."

[private]
_pytest args junit:
    mkdir -p {{reports}}
    {{pytest}} --junitxml={{reports}}/{{junit}} {{args}}

[private]
_bench suite output flags:
    mkdir -p {{bench_reports}}
    {{uv}} run python benchmarks/run_speed.py --suite {{suite}} --output {{bench_reports}}/{{output}} {{flags}}
