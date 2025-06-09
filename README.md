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

## How it works

`simgrep` works in two main modes:

* Quick search. For one-off searches. It indexes files in memory, performs the search, and discards the index.
* Projects. For searching the same set of files repeatedly (like a codebase). It creates a persistent index on disk, allowing for fast subsequent searches and incremental updates.

## Usage examples

### Quick search

This is the fastest way to get started. `simgrep` will build a temporary index for your search path.

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

**1. Create a project:**
First, create a named project for your codebase.

```bash
simgrep project create my-codebase
```

**2. Add paths to the project:**
Tell `simgrep` which directories to include in this project. You can add multiple paths.

```bash
simgrep project add-path ./my-codebase/backend --project my-codebase
simgrep project add-path ./my-codebase/docs --project my-codebase
```

**3. Index your project:**
Build the index. This may take some time for the first run.

```bash
simgrep index --project my-codebase
```

*Note: If `--project` is omitted, `simgrep` uses a `default` project.*

**4. Search your project:**
Now you can search without specifying a path. `simgrep` will use your project's persistent index.

```bash
simgrep search "user session management" --project my-codebase
```

**5. Update the index:**
When you change your files, run `index` again. It will incrementally update the index by only processing new and modified files, which is very fast.

```bash
simgrep index --project my-codebase
```
