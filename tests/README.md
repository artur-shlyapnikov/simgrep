# tests readme

Tests follow hexagonal architecture: unit/domain, unit/application, infrastructure, contract, integration, e2e.

1. domain tests cover pure logic via port interfaces only.
2. application tests drive services using fakes that implement those ports.
3. infrastructure tests verify concrete adapters against port contracts and external deps.
4. contract tests in `tests/contract` assert adapter compliance with shared specs.

Use pytest markers: `@pytest.mark.contract`, `@pytest.mark.external`, `@pytest.mark.slow`.
`conftest.py` exposes only domain-safe fixtures; infra gets its own fixtures.
