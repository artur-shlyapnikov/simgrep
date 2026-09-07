# tests

Current tests cover the simplified local-project engine:

1. `unit/domain`: dataclasses and text/span helpers.
2. `unit/application`: indexing, search, and ranking with fakes.
3. `unit/infrastructure`: config, project TOML, file planning, store, runtime, vector adapter.
4. `integration`: persistent index/search flow with fake runtime.
5. `e2e`: CLI local project and output behavior.
6. `adapter_external`: optional model/native adapter checks.

Use `just test` for the fast suite and `just test-external` for model/native adapter checks.
