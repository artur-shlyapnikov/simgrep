# Contributing

Start with a small issue or pull request describing the behavior you want to change.
For bugs, include the command, expected result, actual result, Python version, and
operating system. Use a minimal synthetic example instead of private source files.

## Local checks

Install Python 3.12+, uv, and just, then run:

```sh
just install
just lint format-check typecheck test
just test-e2e
```

External/model tests may download models and require additional system libraries.
Run `just --list` for focused tests and benchmarks. Add a regression test for a
behavioral change and keep CLI documentation in sync with it.

Contributions are licensed under the repository’s Apache-2.0 license.
