# simgrep

**Work In Progress**

`simgrep` is a command-line tool for semantic search in local files. It aims to find text snippets based on meaning, not just exact keywords.

It's designed for ephemeral (one-off) searches and will soon support persistent project-based indexing.

See the [Architecture Document](docs/architecture.md) for more details.

## Current Capabilities (Ephemeral Search)

When you run `simgrep "your query" ./path/to/search`:

*   **Searches:**
    *   A single file.
    *   Recursively through a directory (files matching `--pattern`; defaults to `*.txt`).
*   **Processing:**
    *   Extracts text from files using `unstructured`.
    *   Chunks text using token-based strategies (configurable model, size, overlap - defaults used for now).
    *   Generates embeddings for your query and text chunks (uses `sentence-transformers`).
    *   Manages chunk metadata (source file, offsets) using an in-memory DuckDB.
*   **Finds & Displays:**
    *   Performs semantic similarity search using an in-memory USearch index.
    *   Outputs results showing the relevant file, similarity score, and the text chunk (`--output show`, default).
    *   Lists unique file paths containing matches (`--output paths`).

### Examples

Search only markdown files:

```bash
simgrep search "apples" docs --pattern "*.md"
```

Search both text and markdown files:

```bash
simgrep search "apples" docs --pattern "*.txt" --pattern "*.md"
```

## Near Future Plans (Persistent Indexing - Phase 3)

*   **Persistent Indexing:**
    *   `simgrep index <path>`: Create or update a persistent index for a specified path (initially for a default project).
    *   Store embeddings in a persistent USearch index on disk.
    *   Store file and chunk metadata in a persistent DuckDB database on disk.
*   **Search Persistent Index:**
    *   `simgrep search "your query"`: Search against the default persistent index without specifying a path.
*   **Configuration & Status:**
    *   Basic global configuration (e.g., for database paths, default model).
    *   `simgrep status`: Show information about the current state of the default index (e.g., number of files/chunks indexed).
*   **Incremental Indexing:**
    *   Efficiently update the index by only processing new or modified files.

(Further phases include named projects, more output modes like RAG, and advanced configuration as outlined in the [Implementation Plan](docs/implementation-plan.md) and [Architecture Document](docs/architecture.md)).