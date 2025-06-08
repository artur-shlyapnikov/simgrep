# simgrep

`simgrep` is a command-line tool for semantic search in local files. It finds text snippets based on meaning, not just exact keywords.

It supports quick one-off searches and persistent indexing for a default project.

See the [Architecture Document](docs/architecture.md) for more details.

## Use Cases
See [use-case diagrams](docs/use-cases.md) for ephemeral search, persistent indexing, and RAG-based answering.


## Current Capabilities (Ephemeral Search)

When you run `simgrep "your query" ./path/to/search`:

*   **Searches:**
    *   A single file.
    *   Recursively through a directory. Use `--pattern` to specify glob(s) for files (defaults to `*.txt`). Pass multiple `--pattern` options to include several patterns.
*   **Processing:**
    *   Extracts text from files using `unstructured`.
    *   Chunks text using token-based strategies (configurable model, size, overlap - defaults used for now).
    *   Generates embeddings for your query and text chunks (uses `sentence-transformers`).
    *   Manages chunk metadata (source file, offsets) using an in-memory DuckDB.
*   **Finds & Displays:**
    *   Performs semantic similarity search using an in-memory USearch index.
    *   Outputs results showing the relevant file, similarity score, and the text chunk (`--output show`, default).
    *   Lists unique file paths containing matches (`--output paths`).
    *   Display a full dependency tree for a code file (`--output imports`).
    *   Limit the number of matches returned with `--top N` (alias `--k`).

### Examples

Search only markdown files:

```bash
simgrep search "apples" docs --pattern "*.md"
```

Search both text and markdown files:

```bash
simgrep search "apples" docs --pattern "*.txt" --pattern "*.md"
```

Limit results to the top 3 matches:

```bash
simgrep search "apples" docs --top 3
```

## Persistent Indexing

*   **Indexing:**
    *   `simgrep index <path>` builds or updates the default project's index for the given path.
    *   Embeddings are stored in a USearch index on disk and metadata in a DuckDB database.
*   **Searching:**
    *   `simgrep search "your query"` searches the default index when no path is provided.
*   **Status:**
    *   `simgrep status` shows how many files and chunks are indexed.
*   **Incremental Updates:**
    *   Only new or changed files are processed on subsequent indexing runs.

(Further phases include named projects, more output modes like RAG, and advanced configuration as outlined in the [Implementation Plan](docs/implementation-plan.md) and [Architecture Document](docs/architecture.md)).