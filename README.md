# simgrep

`simgrep` is an early-stage semantic search CLI for local files. It embeds a query and file
chunks with a local Hugging Face model, then combines semantic similarity with lexical ranking.
It can build a temporary in-memory index for a one-off scan or keep a project index on disk for
repeated searches.

File contents, queries, and embedding inference stay on the local machine. The first index or
ephemeral search can download the configured model from Hugging Face.

The current release is `0.1.0`. This is a greenfield project, so interfaces may change between
releases.

## Prerequisites

- Python 3.12 or newer. The repository selects Python 3.12 in `.python-version`.
- [uv](https://docs.astral.sh/uv/) for the environment and package commands.
- [just](https://just.systems) for the repository's setup, check, test, and benchmark recipes.

## Install

From a checkout, install the runtime package and its locked dependencies:

~~~bash
uv sync --locked
uv run simgrep --version
~~~

For the full developer environment, install the dev and security dependency groups:

~~~bash
just install
# Equivalent to:
uv sync --locked --group dev --group security
~~~

To install the configured embedding model and the NLTK data used by the extraction stack, run:

~~~bash
just setup
~~~

`just setup` checks `.python-version`, installs the locked dev environment, and runs
`just download-models`. Model downloads need network access. You can check or cache only the
configured model with:

~~~bash
uv run simgrep models status
uv run simgrep models cache
~~~

The examples below use `simgrep` as a command on `PATH`. When running from this checkout, use
`uv run simgrep` in the repository directory or prefix commands run elsewhere with
`uv run --project /path/to/simgrep simgrep`.

## Quick start

### One-off search

Pass a path to search without creating a project. The command builds an in-memory index and
discards it when the command exits.

~~~bash
uv run simgrep search "where do failed HTTP requests retry" /path/to/codebase
~~~

When no active project exists, `PATH` is required for a one-off search. Use `--ephemeral` to
force this mode when an active project also covers the path.

### Persistent project search

Initialize a project in the directory you want to search, build its index, and search it:

~~~bash
cd /path/to/codebase
simgrep init
simgrep index
simgrep search "where do failed HTTP requests retry"
~~~

Run `simgrep index` again after adding, editing, or deleting files. The default incremental pass
processes new and changed files and removes files that no longer exist.

If `simgrep` is not installed on `PATH`, run the same workflow from any directory with the
checkout's environment:

~~~bash
uv run --project /path/to/simgrep simgrep -C /path/to/codebase init
uv run --project /path/to/simgrep simgrep -C /path/to/codebase index
uv run --project /path/to/simgrep simgrep -C /path/to/codebase search "where do failed HTTP requests retry"
~~~

## Project indexing

### Initialize and manage paths

~~~text
simgrep init [PATH]
simgrep project add-path PATH
simgrep project remove-path PATH
simgrep project info
~~~

`PATH` defaults to the current directory. `init` creates `.simgrep/project.toml` and registers
the project root as its first indexed path. Use `init --yes` to overwrite an existing project
configuration. `project add-path` accepts another existing path. A path outside the project root
needs `--allow-outside-root`. `project remove-path` cannot remove the last indexed path.

### Build and refresh the index

~~~text
simgrep index
simgrep index --rebuild
simgrep index --dry-run
~~~

The persistent index stores metadata in `.simgrep/metadata.duckdb` and embeddings in
`.simgrep/vectors.usearch`. The default incremental planner compares file size and modification
time. Use `--rebuild` to recreate both index parts and process every discovered file. Use
`--dry-run` to print the number of files that would be indexed without loading the model or
writing index artifacts.

Index options are:

| Option | Effect |
| --- | --- |
| `--rebuild` | Recreate the persistent metadata and embedding indexes. |
| `--dry-run` | Plan the pass without writing or loading the model. |
| `--include GLOB` | Restrict the scan to matching paths. Repeat for more globs. |
| `--exclude GLOB` | Skip matching paths. Repeat for more globs. |
| `--pattern GLOB`, `-p` | Use these scan patterns instead of the configured `file_patterns` for this pass. |
| `--workers N` | Set extraction worker count. The default is `4`. |

### Choose persistent or ephemeral scope

The CLI looks for `.simgrep/project.toml` in the current directory and its parents.

| Invocation | Scope |
| --- | --- |
| `simgrep search QUERY` inside a project | The active persistent project. |
| `simgrep search QUERY PATH` where the project covers `PATH` | The persistent index, filtered to `PATH`. |
| `simgrep search QUERY PATH` outside project coverage | An ephemeral scan of `PATH`. |
| Any of the above with `--persistent` | Require a covering active project. The command fails if none exists. |
| Any of the above with `--ephemeral` | Build a temporary index. Pass `PATH` when no active project exists. |

`similar`, `clusters`, `pack`, and `debt` follow the same persistent-versus-ephemeral selection rules.
`diff`, `expand`, and `rerank` do not use a persistent project index.

Run `simgrep <command> --help` to see every option and its accepted range.

## Search

### Basic search

~~~text
simgrep search QUERY [PATH]
~~~

The default is hybrid ranking. It combines semantic candidates with lexical candidates, returns
five results, and displays the `rich` format. The most useful options are:

| Option | Default or values | Effect |
| --- | --- | --- |
| `--top N`, `--k N` | `5` | Return at most `N` results. |
| `--min-score SCORE` | `0.0`, range `0.0` to `1.0` | Drop results below the score. |
| `--candidates N` | Automatic | Set the semantic candidate pool. Without an explicit value, simgrep uses `max(top * 40, 200)`, or `max(top * 120, 1000)` when scope or path filters are active. |
| `--hybrid`, `--no-hybrid` | Hybrid enabled | Include or disable lexical ranking. |
| `--lexical-top N` | `50` when hybrid ranking is enabled | Set the lexical candidate count. |
| `--lexical-weight WEIGHT` | `0.25` when hybrid ranking is enabled | Set the lexical contribution from `0.0` to `1.0`. |
| `--lexical-fallback MODE` | Accepted values `off`, `fill`, `empty` | Select what to do with lexical-only hits. The current CLI ignores this selection and uses `fill`. |
| `--diversity MODE` | `window` | Choose `window`, `file`, `package`, or `none`. |
| `--freshness MODE` | Global `freshness`, normally `auto` | Choose `auto`, `skip`, or `check` for a persistent index. |
| `--file-filter GLOB` | None | Filter result files. |
| `--keyword TEXT` | None | Keep results whose chunk contains this text. |
| `--include GLOB` | None | Include matching paths. |
| `--exclude GLOB` | None | Exclude matching paths. |
| `--pattern GLOB`, `-p` | Configured patterns | Set scan patterns for an ephemeral search. |
| `--prefer GLOB` | None | Add a path boost. `--prefer-weight` defaults to `0.15`. |
| `--context N`, `-c` | `0` | Add `N` lines before and after each hit. |
| `--max-chars N` | Global `max_chars`, normally `1200` | Truncate displayed snippets. |
| `--why` | Off | Include the per-hit score breakdown. |
| `--whole-unit` | Off | Expand each hit to its enclosing function, class, brace block, or paragraph. |
| `--rerank` | Off | Rerank the first `--rerank-top` hits with a cross-encoder. |
| `--rerank-model NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Choose the model used by `--rerank`. |
| `--rerank-top N` | `25` | Set the number of hybrid-ordered hits passed to the reranker. |

`--diversity window` skips a file when it appeared in the previous two selected results.
`file` keeps one result per file. `package` keeps at most two results per parent directory.
`none` keeps the ranked results without this filtering.

`--include` and `--exclude` filter an ephemeral scan and filter results from a persistent index.
`--pattern` replaces the configured scan patterns for an ephemeral search. `--file-filter` matches
the stored result path or file name. `--keyword` performs a case-insensitive substring check on
the chunk text. `--prefer` adds a path boost; its weight defaults to `0.15` and must be from `0.0`
through `1.0`.

For a persistent project, `--freshness auto` rebuilds a missing index and incrementally updates a
stale one before searching. `skip` trusts the current artifacts. `check` fails instead of updating
when the planner finds a mutation.

### Boolean semantic expressions

Use `--expr` instead of the positional query for per-leaf semantic scoring:

~~~bash
simgrep search --expr '(auth OR login) AND NOT oauth'
simgrep search --expr '"connection pool" AND retry' --format jsonl
~~~

Operators must be uppercase. Lowercase `and`, `or`, and `not` are ordinary words. Adjacent atoms
also form an implicit `AND`. Parentheses group expressions. Expression search is semantic only,
so `--lexical-top` and `--lexical-weight` are rejected. Do not pass both a positional query and
`--expr`.

### Output formats

~~~text
simgrep search "connection pool" --format rich
simgrep search "connection pool" --format compact
simgrep search "connection pool" --format paths
simgrep search "connection pool" --format json
simgrep search "connection pool" --format jsonl
simgrep search "connection pool" --format count
simgrep search "connection pool" --format grep
~~~

`rich` and `compact` are human-readable. `paths` prints unique sorted paths. `count` prints one
result count. `grep` prints `path:line[:score]: snippet`. `json` prints one array and `jsonl`
prints one result object per line. Machine formats keep the payload on stdout and suppress index
progress there. By default a search record contains `path`, `score`, character offsets, `text`,
`stale_offsets`, and line fields when available. `--why` adds the score breakdown. `--no-scores`,
`--no-line-numbers`, and `--absolute-paths` change the human and machine projections where the
format supports them.

## Other commands

### Find similar code or text

~~~text
simgrep similar SOURCE [TARGET_DIR]
~~~

`SOURCE` can be a literal string, `@PATH` for a whole file, `PATH:LINE`, `PATH:START-END`, or `-`
for stdin. `TARGET_DIR` sets the optional corpus path. The command uses an ephemeral scan when
`--ephemeral` is set or the active project does not cover that path. Without it, the command uses
the active project when one exists.

~~~bash
simgrep similar @src/errors/http.py
simgrep similar src/errors/http.py:1-40 --unlike @src/retry/pattern.py --unlike-weight 0.7
cat snippet.go | simgrep similar - ./src --format jsonl
~~~

`--unlike` subtracts similarity to the second anchor from the first. A candidate can then fall
below `--min-score` and disappear. The score is `s_like - lambda * s_unlike`;
`--unlike-weight` defaults to `0.5` and accepts values from `0.0` through `1.0`. File or span
anchors exclude overlapping chunks from their own file unless `--include-self` is set. `similar` also supports
`--scope`, `--file-filter`, `--include-glob`, `--exclude-glob`, `--diversity`, `--freshness`,
`--top`, `--min-score`, and `--format`.

### Find semantic duplicates

~~~bash
simgrep clusters [PATH] --threshold 0.9 --min-size 3
simgrep clusters --format paths
~~~

`clusters` groups chunks whose cosine similarity reaches `--threshold`. It reports cross-file
clusters by default. Add `--same-file` to include duplicates from one file. The defaults are a
threshold of `0.8`, a minimum cluster size of `2`, at most `20` shown clusters, and a `50000`
chunk safety cap. Formats are `rich`, `compact`, `paths`, `json`, `jsonl`, and `count`.

### Compare two trees

~~~bash
simgrep diff OLD_PATH NEW_PATH
simgrep diff ./old ./new --threshold 0.9 --top 10 --format count
~~~

`diff` builds temporary corpora for both operands and matches chunks one-to-one by semantic
similarity. `--threshold` defaults to `0.8`, `--top` limits listed added and removed chunks, and
`--max-chunks` defaults to `50000`. Formats are `rich`, `json`, `jsonl`, and `count`. The command
does not use project configuration or a persistent index.

### Expand a hit

~~~bash
simgrep expand src/payment.py 42
simgrep expand src/payment.py 42 --format json
simgrep search "gateway timeout" ./src --whole-unit --format json
~~~

`expand PATH LINE` needs no index. It uses indentation for Python and YAML, braces for supported
brace languages, and a contiguous non-blank paragraph for other files. Use `--language dedent`,
`--language brace`, or `--language paragraph` to override the family. `--max-chars` truncates the
expanded text and adds a marker. The implementation uses lexical heuristics, not an AST.

### Assemble a context block

~~~bash
simgrep pack "gateway timeout" "refund flow" ./payment-service --budget 3000 --format markdown
~~~

`pack` searches each query, merges duplicate chunks, and selects chunks under the token budget.
The default budget is `3000`, the per-query pool is `8`, and `--lam` defaults to `0.7` for the
relevance-versus-diversity tradeoff. Formats are `rich`, `markdown`, and `json`. The last argument
is treated as `TARGET_DIR` when it names an existing directory and at least one query remains.

### Audit debt markers

~~~bash
simgrep debt ./src --top 10 --format json
simgrep debt . --max-age 90
~~~

`debt` scans uppercase `TODO`, `FIXME`, `XXX`, `HACK`, and `WORKAROUND` markers, groups their
chunks into themes, and reports file ages from git when available. The default threshold is `0.8`,
the minimum theme size is `2`, and the report shows at most `20` themes and `8` matches per theme.
`--max-age DAYS` makes the command fail with exit code `1` when a dated theme is older than the
limit. It also needs at least one available git age for a corpus containing markers. Formats are
`rich`, `json`, and `jsonl`.

### Rerank an external file list

~~~bash
grep -rl "except Exception" src/ | simgrep rerank "swallowing errors silently" --files-from -
simgrep rerank "retry with backoff" src/net.py src/worker.py --format jsonl
simgrep search "retry with backoff" ./src --rerank --rerank-top 25
~~~

`rerank` reads explicit files or paths from `--files-from PATH`. Use `--files-from -` for stdin.
It chunks those files, scores query-document pairs with a local cross-encoder, and returns the
best chunk for each file. It does not need a project index. The default model is
`cross-encoder/ms-marco-MiniLM-L-6-v2`, and the default cap is `512` chunks. The cross-encoder
may download its model on first use.

### Use the interactive REPL

~~~bash
simgrep repl
~~~

`repl` requires an active project. It reuses the runtime while you enter queries. An empty query
or EOF exits the session.

### Inspect and reset a project

~~~bash
simgrep status
simgrep doctor
simgrep reset --yes
~~~

`status` prints the active project's file count, chunk count, and index state. `doctor` checks
the package version, global config, configured model, cache status, and project index. `reset`
deletes local index artifacts while keeping `.simgrep/project.toml`; it asks for confirmation
unless `--yes` is provided.

### Run the MCP stdio server

~~~bash
simgrep mcp
~~~

The server reads newline-delimited JSON-RPC 2.0 requests from stdin and writes protocol responses
to stdout. It exposes `search`, `similar`, `clusters`, `status`, `index`, `diff`, `expand`, `pack`,
and `debt`. For Claude Code, the documented registration command is:

~~~bash
claude mcp add simgrep -- simgrep mcp
~~~

Use the checkout-prefixed command if `simgrep` is not on `PATH`.

## Configuration

Global settings live in `~/.config/simgrep/config.toml`. The file is created when simgrep first
loads the configuration.

~~~bash
simgrep config list
simgrep config get lexical_weight
simgrep config set lexical_weight 0.5
simgrep config set file_patterns "*.py,*.md,*.rst"
~~~

`config set file_patterns` takes a comma-separated list. A project configuration in
`.simgrep/project.toml` stores the project name, indexed paths, model, `chunk_size`, and
`chunk_overlap`. `simgrep init` copies the global model and chunk settings into a new project.
Edit the project file and rebuild the index when a project needs different model or chunking
settings.

Default global values are:

| Key | Default |
| --- | --- |
| `model` | `ibm-granite/granite-embedding-30m-english` |
| `chunk_size` | `128` tokens |
| `chunk_overlap` | `20` tokens |
| `batch_size` | `128` |
| `max_file_size_bytes` | `10485760` bytes, or 10 MiB |
| `follow_symlinks` | `false` |
| `file_patterns` | The scan list below. `config set` accepts comma-separated globs. |
| `lexical_top` | `50` |
| `lexical_weight` | `0.25` |
| `freshness` | `auto` |
| `context_lines` | `0` (stored; use `--context` to control search output) |
| `max_chars` | `1200` |

The default scan patterns are:

~~~text
*.txt *.md *.rst *.py *.js *.ts *.tsx *.jsx *.java *.go *.rs *.c *.cpp *.h *.hpp *.cs
*.rb *.php *.swift *.kt *.scala *.sh *.bash *.zsh *.toml *.yaml *.yml *.json *.xml *.html
*.css *.sql *.dockerfile Dockerfile
~~~

The `SIMGREP_DEVICE` environment variable can force `cpu`, `mps`, or `cuda`. Without it, the
embedder chooses CUDA, then MPS, then CPU. Set `SIMGREP_EMBED_RUNTIME=torch` to force the torch
query path instead of the automatic ONNX query path.

## Files and privacy

The scanner respects `.gitignore` and `.repo_ignore`. It skips common build and cache directories,
symbolic links by default, binary content, and sensitive filenames such as `.env`, private keys,
and kubeconfig files. It also skips files larger than `max_file_size_bytes` unless that setting is
changed.

`.simgrep/` contains extracted metadata and embeddings derived from indexed files. Keep it out of
version control unless you have a reason to share the index. Review ignore rules before indexing
credentials or other private data. The first model download connects to Hugging Face; cached
inference runs locally. The MCP server gives its host process access to the configured search
commands over stdin and stdout.

## Known limitations

- The incremental planner uses file size and modification time. If an edit leaves both values
  unchanged, run `simgrep index --rebuild`.
- A narrowed `--include`, `--exclude`, or `--pattern` scan can make previously indexed files look
  deleted. Run `simgrep index --dry-run` before applying such a scan.
- `--lexical-fallback` is exposed by the CLI, but the current search handler does not forward the
  selected value. The effective behavior remains the default `fill` mode. The implementation's
  enum values are `off`, `fill`, and `empty`.
- `expand` uses lexical heuristics. It does not parse an AST, and brace detection does not model
  template or regular-expression literals.
- `clusters` and `diff` cap the processed corpus at `50000` chunks by default. Clustering compares
  chunk pairs, so narrow the path or raise `--max-chunks` for larger work.
- `diff` matches content by semantic similarity, not by git rename or move history. An unchanged
  move or rename is not reported as a content change.
- `pack` estimates tokens as approximately one token per four characters. The estimate can differ
  from the tokenizer used by a downstream model.
- The real-model benchmark suite needs a cached model. The stress benchmark suite is not yet
  implemented.

## Development

The repository uses Ruff, strict Mypy, and pytest. The main recipes are:

~~~bash
just install
just run                         # runs simgrep --help
just lint
just format-check
just typecheck
just test                        # unit and integration, without external or slow tests
just test-e2e                    # CLI tests
just test-external               # model and native adapter tests
just test-all                    # full suite with coverage
just check                       # lint, format-check, typecheck, test, and test-e2e
~~~

`just format` applies Ruff formatting. `just security` runs dependency, workflow, secret, and
Bandit checks. The secret scan needs `gitleaks` on `PATH`. Use `just --list` for every recipe.

The benchmark recipes are:

~~~bash
just bench-speed
just bench-speed-record
just bench-speed-real
~~~

The CI benchmark suite uses deterministic fake embeddings. The real-model suite needs the
configured model in the local cache. See [benchmarks/README.md](benchmarks/README.md) for the
benchmark harness and report format.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the local contribution checks and
[docs/use-cases.md](docs/use-cases.md) for longer command examples.

## License

[Apache-2.0](LICENSE).
