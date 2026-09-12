# Use cases

This page collects workflows that need more context than the command reference in the main
[README](../README.md).

## Run against another repository

When the `simgrep` checkout is not installed globally, run its CLI with `uv` and point it at the
repository to search:

```bash
uv run --project /path/to/simgrep simgrep init /path/to/target-repo
uv run --project /path/to/simgrep simgrep -C /path/to/target-repo index
uv run --project /path/to/simgrep simgrep -C /path/to/target-repo search "invoice status transitions"
```

`-C` and `--project-root` change the process working directory before the command runs. The active
project lookup then starts in the target repository.

## Choose persistent or ephemeral search

Create a project when you search the same paths more than once:

```bash
cd /path/to/target-repo
simgrep init
simgrep index
simgrep search "rollback payment"
```

The index stores metadata in `.simgrep/metadata.duckdb` and vectors in `.simgrep/vectors.usearch`.
The next `simgrep index` pass checks for new, changed, and deleted files.

Use an ephemeral search when you do not want project files or index artifacts:

```bash
simgrep search "rollback payment" ./src --ephemeral
```

Passing a path outside the active project's indexed paths also selects an ephemeral scan. Use
`--persistent` when the command must fail instead of falling back. A search without a path still
needs an active project.

## Control stale indexes

Persistent reads use the global `freshness` setting, which defaults to `auto`:

```bash
simgrep search "rollback payment" --freshness auto
simgrep search "rollback payment" --freshness check
simgrep search "rollback payment" --freshness skip
```

`auto` updates a stale index before the search. `check` fails when files changed. `skip` trusts the
stored index. `simgrep index --dry-run` previews the current index pass before it writes new vectors.

## Manage indexed paths

```bash
simgrep project add-path docs
simgrep project add-path ../shared --allow-outside-root
simgrep project remove-path docs
simgrep project info
```

The project must keep at least one indexed path. Paths inside the project root are stored relative
to that root when possible.

## Use machine-readable output

Search supports `json`, `jsonl`, `paths`, `count`, and `grep` in addition to the human-readable
`rich` and `compact` formats:

```bash
simgrep search "tax invoice" --format json
simgrep search "tax invoice" --format jsonl
simgrep search "tax invoice" --format paths
simgrep search "tax invoice" --format count
simgrep search "tax invoice" --format grep
```

The JSON format returns one array. JSONL returns one result object per line. `paths` prints unique
paths, `count` prints the number of result objects, and `grep` prints `path:line[:score]: snippet`.
Machine formats reserve stdout for the payload. Progress and warnings go to stderr.

JSON search records include `path`, `score`, `start_char`, `end_char`, `text`, and
`stale_offsets`. Records can also include `line_start`, `line_end`, `context_before`,
`context_after`, and `why` when the corresponding options produce those fields.

## Filter and diversify search results

```bash
simgrep search "retry logic" ./src --ephemeral --include "*.py" --exclude "tests/**"
simgrep search "retry logic" ./src --pattern "*.py"
simgrep search "retry logic" --file-filter "*.py" --keyword timeout
simgrep search "config loading" --diversity file
```

`--include` and `--exclude` filter the current ephemeral scan and persistent results. `--pattern`
replaces the default scan patterns for an ephemeral search. It does not change a persistent index.
`--file-filter` matches the stored result path or its file name. `--keyword` performs a
case-insensitive substring check on the chunk text.

The `--diversity` modes have these limits:

- `window` avoids the same file in the two most recent selected results.
- `file` keeps at most one result per file.
- `package` keeps at most two results per parent directory.
- `none` applies no path diversity filter.

## Find similar text with an anchor

`similar` accepts a literal anchor, a whole file, a line range, or stdin:

```bash
simgrep similar @./src/errors/http.py
simgrep similar ./src/errors/http.py:1-40
cat snippet.go | simgrep similar - ./src --format jsonl
```

Use `--unlike` to subtract a second anchor from the first:

```bash
simgrep similar ./src/errors/http.py:1-40 \
  --unlike @./src/retry/pattern.py --unlike-weight 0.7
```

The target directory sets the optional corpus path. The command uses an ephemeral scan when
`--ephemeral` is set or the active project does not cover the path. The combined semantic score is
`like - weight * unlike`. A candidate can fall below `--min-score` after the subtraction. The
weight defaults to `0.5` and must stay between `0.0` and `1.0`. A file or line-range anchor
excludes overlapping chunks from its own file unless `--include-self` is set. The MCP server cannot
accept `-` as an anchor because stdin carries the JSON-RPC protocol there.

## Combine semantic query terms

Use `--expr` for boolean semantic search:

```bash
simgrep search --expr "(auth OR login) AND NOT oauth"
simgrep search --expr '"connection pool" AND retry' --format jsonl
```

`AND`, `OR`, and `NOT` must be uppercase. Quoted phrases are single leaves and parentheses group
the expression. Adjacent terms use an implicit `AND`. Expressions use semantic scores only. The
CLI rejects `--lexical-top` and `--lexical-weight` with `--expr`, and it does not accept a
positional query at the same time.

## Find duplicate chunks

```bash
simgrep clusters
simgrep clusters ./vendor --threshold 0.9 --min-size 3
simgrep clusters --format json
```

Without a path, the command reads the active project. A path outside project coverage triggers an
ephemeral scan. Clusters use pairwise similarity and default to cross-file matches. Add
`--same-file` to include duplicates within one file. Clusters sort by their duplicated line count.
The `--max-chunks` guard defaults to `50000` because the comparison grows quadratically.

## Compare two trees

```bash
simgrep diff ./release-1.2 ./release-1.3
simgrep diff ./old ./new --threshold 0.9 --top 10
simgrep diff ./old ./new --format json
```

`diff` creates temporary corpora for both operands, which can each be a file or directory. It
matches chunks greedily, one-to-one, at or above the threshold. An unchanged file move or rename can
match across paths and therefore does not appear as an added or removed chunk. Use `--format count`
for a summary string such as `12 matched, 2 added, 1 removed`. The default `--max-chunks` is
`50000` across both trees.

## Group debt markers

```bash
simgrep debt . --top 10 --format json
simgrep debt . --max-age 90
```

The scanner recognizes uppercase `TODO`, `FIXME`, `XXX`, `HACK`, and `WORKAROUND` markers. It
groups their chunks by vector similarity and reports theme labels, members, and the oldest Git
commit date available for each theme.

`--max-age` fails when a dated theme is older than the given number of days. Run it in a Git
repository. Untracked or non-Git files have no age. If no marker file has a Git age, the command
cannot evaluate the age gate. A scan with no markers exits successfully.

## Expand a hit to a larger unit

Search returns chunks. Use `--whole-unit` when the result needs the enclosing function, class, block,
or paragraph:

```bash
simgrep search "gateway timeout rollback" ./payment-service --whole-unit --format json
```

For a single file and line, use `expand` without an index:

```bash
simgrep expand src/payment.py 42
simgrep expand src/payment.py 42 --format json
simgrep expand notes.md 3 --language paragraph
```

The command chooses `dedent`, `brace`, or `paragraph` by extension. `--language` accepts those
family names and overrides the choice. The implementation uses lexical heuristics rather than an
AST parser. The brace scanner ignores braces in ordinary strings and comments, but it does not
model template or regular-expression literals.

## Assemble a budgeted context block

`pack` runs one search per query, deduplicates the result pool by chunk label, and selects chunks
until the estimated budget is full:

```bash
simgrep pack "gateway timeout" "refund flow" ./payment-service \
  --budget 3000 --format markdown
simgrep pack "gateway timeout" "refund flow" --format json
```

The last argument is a target directory only when it is an existing directory. `--budget` defaults
to `3000`, `--per-query` to `8`, and `--lam` to `0.7`. The token estimate is
`ceil(character_count / 4)`, so it is not a model-token count. The budget applies to this estimate.

## Rerank candidates from another tool

Use `rerank` after `grep`, `rg`, or another tool has produced candidate files:

```bash
rg -l "except Exception" src/ | \
  simgrep rerank "swallowing errors silently" --files-from -
simgrep rerank "retry with backoff" src/net.py src/worker.py --format jsonl
```

The command reads and chunks the candidate files, scores their chunks with the default local
cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`, and returns the best chunk per file. It does
not read a simgrep index or run vector search. `--files-from` accepts `-` for stdin or a file with
one path per line. `--max-chunks` defaults to `512`.

## Connect an MCP client

Start the stdio server with:

```bash
simgrep mcp
```

The server reads one JSON-RPC 2.0 request per line from stdin and writes responses for requests to
stdout. It exposes `search`, `similar`, `clusters`, `status`, `index`, `diff`, `expand`, `pack`, and
`debt`. It does not open a network listener.

For a client that supports Claude's MCP registration command:

```bash
claude mcp add simgrep -- simgrep mcp
```

## Cache models and inspect configuration

```bash
simgrep models status
simgrep models cache ibm-granite/granite-embedding-30m-english
simgrep config list
simgrep config get model
simgrep config set lexical_top 25
simgrep doctor
```

`models cache` downloads the selected Hugging Face model for later offline-first loads. The
`config` commands read and write `~/.config/simgrep/config.toml`. `doctor` prints the version,
config status, model cache status, active project, and index counts.
