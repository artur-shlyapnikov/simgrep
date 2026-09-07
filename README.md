# simgrep

Semantic grep for local files. `simgrep` finds text by **meaning**, not keywords — ask for
"database connection errors" and it surfaces `pool exhausted` or `timeout connecting to postgres`,
even when the query words do not appear in the file.

It embeds your query and files into a shared vector space and ranks hits by fusing semantic
similarity with lexical scores. Embeddings run locally via sentence-transformers. File contents and search queries are
processed on your machine; the initial model download connects to Hugging Face.

- **No setup for one-off searches** — ephemeral in-memory index, built and discarded per query.
- **Persistent projects** — incremental on-disk index for codebases you search repeatedly.
- **Hybrid ranking** — dense vectors plus lexical tables, with path boosts and result diversity.
- **Whole units, not windows** — expand any hit to its enclosing function/class/block (`expand` / `--whole-unit`).
- **Beyond search** — clones (`clusters`), semantic diff (`diff`), debt themes (`debt`), context packs (`pack`), cross-encoder rerank (`rerank`), query-by-example (`similar`).

```console
$ simgrep search "give customers their money back after a failed payment" ./payment-service
src/payment/controller/PaymentController.java:20-31  score=0.805
    public RollbackResult rollbackPayment(@PathVariable Long id) {
```

**Requirements:** Python 3.12+ (`./.python-version`), [uv](https://docs.astral.sh/uv/) and [just](https://just.systems).

## Quick start

```bash
just setup            # check python, install deps, pre-download the embedding model
just run search "where do we retry failed HTTP requests" ./src

# or without just:
uv sync --group dev --group security
uv run simgrep search "where do we retry failed HTTP requests" ./src
```

First use downloads the embedding model (~100 MB) from Hugging Face Hub;
`just setup` / `just download-models` pre-caches it. Models load offline-first from the local cache.

For a directory you search repeatedly, build a persistent index instead:

```bash
cd /path/to/my-codebase
simgrep init          # creates .simgrep/project.toml
simgrep index         # incremental; --rebuild to start over, --dry-run to preview
simgrep search "user session management"
simgrep index         # re-run after editing files (only new/changed files processed)
```

With a `PATH` argument, persistent search is used when an indexed project covers the path,
otherwise it falls back to ephemeral. `--persistent` fails fast without a covering project;
`--ephemeral` always re-indexes on the fly. Add `.simgrep/` to `.gitignore` unless you want to
version the index.

## Commands

| Command | What it does |
|---|---|
| `search QUERY [PATH]` | Semantic (+lexical) search; `--expr "(auth OR login) AND NOT oauth"` for boolean queries |
| `similar SOURCE [DIR]` | Query-by-example: `@file`, `path:start-end`, `-` (stdin), or literal text; `--unlike` for contrast |
| `clusters [PATH]` | Group semantically duplicated chunks (copy-paste clones, drifted variants) |
| `diff OLD NEW` | Semantic tree diff: what appeared/disappeared; renames and moves are invisible |
| `expand FILE LINE` | Expand a line to its enclosing unit (function/class/brace block/paragraph) |
| `pack Q... [DIR]` | Assemble queries into one paste-ready context block under a token budget (`--budget`, `--lam`) |
| `debt [PATH]` | Cluster `TODO`/`FIXME`/`XXX`/`HACK`/`WORKAROUND` markers into themes with git ages; `--max-age N` CI gate |
| `rerank QUERY FILES...` | Score external candidates with a local cross-encoder; `search --rerank` reranks top hits in place |
| `repl` | Interactive search (model and index stay loaded) |
| `init`, `index`, `status`, `doctor`, `reset` | Project lifecycle: create, build, inspect, sanity-check, delete index artifacts |
| `project add-path\|remove-path\|info` | Cover more paths, stop covering one, show root and covered paths |
| `models status\|cache`, `config list\|get\|set` | Model cache and `~/.config/simgrep/config.toml` management |
| `mcp` | Dependency-free MCP stdio server for agents (`claude mcp add simgrep -- simgrep mcp`) |

Useful flags:

```bash
simgrep search "async helpers" ./src --pattern "*.py" --top 3 --format paths
simgrep search "connection pool" ./configs --no-hybrid --min-score 0.5 --context 2
simgrep search "gateway timeout" ./src --whole-unit --why --diversity file
simgrep search "auth" ./src --include "*.py" --exclude "*.gen.*" --prefer "src/auth/**"
simgrep search "auth middleware" ./src --format json        # rich|compact|paths|json|jsonl|count|grep
simgrep clusters ./src --threshold 0.9 --format jsonl | jq '.score'
simgrep diff ./old ./new --threshold 0.9 --format count
simgrep debt . --max-age 90                                 # exit 1 when a dated theme is older
```

Run `simgrep <command> --help` for full options.

## Configuration

Defaults live in `~/.config/simgrep/config.toml` (auto-created); projects can override
model/chunking in `.simgrep/project.toml`:

```bash
simgrep config list
simgrep config set chunk_size 256
```

Defaults: model `ibm-granite/granite-embedding-30m-english`, `chunk_size=128`,
`chunk_overlap=20`, `lexical_top=50`, `lexical_weight=0.25`, `max_file_size_bytes=10485760`
(10 MB), `follow_symlinks=false`. `SIMGREP_DEVICE` (`cpu`, `mps`, `cuda`) overrides device
selection.

Scanned by default: `.txt .md .rst .py .js .ts .tsx .jsx .java .go .rs .c .cpp .h .hpp .cs
.rb .php .swift .kt .scala .sh .bash .zsh .toml .yaml .yml .json .xml .html .css .sql`,
`*.dockerfile`, `Dockerfile`. Other text-like formats fall back to `unstructured` extraction.
Binaries are skipped. Respects `.gitignore`; extra ignores go in `.repo_ignore`.

## Project structure

```text
simgrep/main.py        Typer CLI (all commands above)
simgrep/search.py, simgrep/indexing.py, simgrep/store.py   search / indexing / DuckDB+usearch store
simgrep/clusters_engine.py, simgrep/diff_engine.py, simgrep/debt_engine.py, simgrep/pack.py, simgrep/rerank.py, simgrep/expand.py
simgrep/adapters/      embedder (sentence-transformers/ONNX), chunker, extractor, vector, reranker
simgrep/models.py, simgrep/config.py, simgrep/project.py, simgrep/mcp_server.py
tests/unit|integration|e2e|adapter_external|fixtures/
docs/                  use-cases.md, architecture-decision-records/
benchmarks/ scripts/   speed suites, HF model pre-cache
justfile               setup, check, test, bench, run recipes
```

## Development

```bash
just install         # editable install + dev/security deps
just lint            # ruff (E, F, I)
just format          # ruff format
just typecheck       # strict mypy on simgrep/ and tests/
just test            # unit + integration (excludes external/slow)
just test-e2e        # CLI behavior
just test-all        # full suite with coverage
just security        # pip-audit, bandit, zizmor, gitleaks
```

Design background: [docs/use-cases.md](docs/use-cases.md).

## Status

`v0.1.0`, early development. Greenfield project: interfaces may break between releases.

## Privacy and security

Indexes contain extracted file content and should be treated as sensitive local data.
Keep `.simgrep/` out of version control, review your ignore rules before indexing,
and avoid indexing credentials or private data you do not need to search.
Model downloads require network access; cached embedding inference runs locally.
The MCP server uses stdio and exposes search capabilities to its host process.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contributions and
[SECURITY.md](SECURITY.md) for vulnerability reporting.

## License

[Apache-2.0](LICENSE).
