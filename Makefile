.PHONY: install lint format format-check test test-unit test-property test-integration test-e2e typecheck run clean help download-models setup check-python-version all

PYTHON_VERSION := $(shell cat .python-version)
UV := uv

help:
	@echo "Makefile for simgrep development"
	@echo ""
	@echo "Available targets:"
	@echo "  install          - Install dependencies and the project in editable mode"
	@echo "  lint             - Run linters (Ruff check)"
	@echo "  format           - Format code (Ruff format)"
	@echo "  format-check     - Check code formatting (Ruff format --check)"
	@echo "  test             - Run all tests (Pytest with coverage)"
	@echo "  test-unit        - Run unit tests"
	@echo "  test-property    - Run property tests"
	@echo "  test-integration - Run integration tests"
	@echo "  test-e2e         - Run end-to-end tests"
	@echo "  typecheck        - Run static type checker (Mypy)"
	@echo "  download-models  - Pre-download and cache Hugging Face models used by the project"
	@echo "  run              - Run the simgrep application (e.g., simgrep --help)"
	@echo "  clean            - Clean up build artifacts and cache files"
	@echo ""

install:
	@echo "Installing dependencies and project in editable mode..."
	$(UV) pip install -e .[dev]

lint:
	@echo "Running linter (Ruff check)..."
	$(UV) run ruff check .

format:
	@echo "Formatting code (Ruff format)..."
	$(UV) run ruff format .

format-check:
	@echo "Checking code formatting (Ruff format --check)..."
	$(UV) run ruff format --check .

test:
	@echo "Running all tests (Pytest with coverage)..."
	mkdir -p reports
	$(UV) run pytest -n auto \
	    --junitxml=reports/junit.xml \
	--cov=simgrep --timeout=30 tests/

test-unit:
	@echo "Running unit tests..."
	mkdir -p reports
	$(UV) run pytest -n auto --junitxml=reports/junit-unit.xml --cov=simgrep tests/unit/ --timeout=10

test-contract:
	@echo "Running contract tests..."
	mkdir -p reports
	$(UV) run pytest -n auto --junitxml=reports/junit-contract.xml --cov=simgrep -m contract --timeout=20

test-property:
	@echo "Running property tests..."
	mkdir -p reports
	$(UV) run pytest -n auto --junitxml=reports/junit-property.xml --cov=simgrep 'tests/unit/infrastructure/test_usearch_index_property.py' 'tests/unit/infrastructure/test_hf_chunker_property.py' --timeout=20

test-integration:
	@echo "Running integration tests..."
	mkdir -p reports
	$(UV) run pytest -n auto --junitxml=reports/junit-integration.xml --cov=simgrep tests/integration/ --timeout=20

test-e2e:
	@echo "Running end-to-end tests..."
	mkdir -p reports
	$(UV) run pytest -n auto --junitxml=reports/junit-e2e.xml --cov=simgrep tests/e2e/ --timeout=30

typecheck:
	@echo "Running static type checker (Mypy)..."
	$(UV) run mypy simgrep/ tests/

download-models:
	@echo "Downloading and caching Hugging Face models..."
	$(UV) run python scripts/cache_hf_model.py

run:
	@echo "Running simgrep --help as an example..."
	$(UV) run simgrep --help

clean:
	@echo "Cleaning up..."
	find . -type f -name '*.py[co]' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .venv # If uv creates a .venv by default and it's not globally managed
	@echo "Cleaned."

check-python-version:
	@if [ ! -f .python-version ]; then \
		echo "Error: .python-version file not found. Please create it with your Python version (e.g., 3.12)."; \
		exit 1; \
	fi
	$(UV) version # A simple uv command to ensure it's working with the environment

setup: check-python-version install download-models
	@echo "Setup complete."

all: install lint format-check test typecheck
	@echo "All checks passed."