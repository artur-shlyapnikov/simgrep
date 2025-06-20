# simgrep

`simgrep` is a command-line tool for semantic search in local files. It finds text snippets based on meaning, not just exact keywords.

## Installation

From source:

```bash
# recommended: use uv to create a virtual environment
uv venv
source .venv/bin/activate

# Install the package
uv pip install .
```

For development, use `make install` to install in editable mode with dev dependencies.

## First-Time Setup

Before using persistent projects, run the one-time global setup:

```bash
simgrep init --global
```

This creates configuration files and databases in your home directory (`~/.config/simgrep/`).

## How it works

`simgrep` works in two main modes:

* Quick search. For one-off searches. Results are cached under `~/.cache/simgrep/quicksearch` so repeated searches on the same path are faster. Use `simgrep clean-cache` to remove these files.
* Projects. For searching the same set of files repeatedly (like a codebase). It creates a persistent index on disk, allowing for fast subsequent searches and incremental updates.

## Usage examples

### Quick search

This is the fastest way to get started. `simgrep` caches quick search indexes under `~/.cache/simgrep/quicksearch` so repeated searches are faster. Use `simgrep clean-cache` to clear the cache.

**Search a directory for a concept:**
Find text related to "database connection errors" in your project's `src` folder.

```bash
simgrep search "database connection errors" ./src
```

**Filter by file type:**
Search only within Python files. Use `--pattern` for glob patterns.

```bash
simgrep search "async function examples" ./my_project --pattern "*.py"
```

You can specify multiple patterns:

```bash
simgrep search "api documentation" ./docs --pattern "*.md" --pattern "*.rst"
```

**Change output format:**
Instead of showing matching text, just list the files that contain matches.

```bash
simgrep search "user authentication flow" ./docs --output paths
```

**Limit the number of results:**
Get the top 3 most relevant results. Use `--top` or its alias `--k`.

```bash
simgrep search "database connection pool" ./configs --top 3
```

### Working with Projects

Use projects to create a reusable index for a directory you search often, like a large codebase or your notes. This is much faster for subsequent searches.

**1. Initialize a project:**
Navigate to your project's root directory and run:

```bash
cd /path/to/my-codebase
simgrep init
```

This creates a `.simgrep` directory, registers a new project (e.g., `my-codebase`), and automatically adds the current directory as a path to be indexed.

**2. Add more paths to the project (optional):**
If your project has multiple source directories (e.g., `backend` and `docs`), you can add them. `simgrep` will automatically use the project context from your current directory.

```bash
# from /path/to/my-codebase
simgrep project add-path ./backend
simgrep project add-path ./docs
```

**3. Index your project:**
Build the index. This may take some time for the first run.

```bash
simgrep index
```

### Indexing with Multiple Workers

You can speed up indexing by enabling concurrent file processing. Use the
`--workers` option to control the number of worker threads:

```bash
simgrep index --workers 4
```

By default, `simgrep` uses all available CPU cores.

*Note: `simgrep` automatically detects you are in the `my-codebase` project. You can also be explicit with `--project my-codebase`.*

**4. Search your project:**
Now you can search without specifying a path. `simgrep` will use your project's persistent index.

```bash
simgrep search "user session management"
```

**5. Update the index:**
When you change your files, run `index` again. It will incrementally update the index by only processing new and modified files, which is very fast.

```bash
simgrep index
```