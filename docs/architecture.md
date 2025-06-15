## Semantic Grep (`simgrep`) - Design Document

**Version:** 1.0
**Date:** May 14, 2025
**Status:** Proposed

**Abstract:**
`simgrep` is a command-line utility that enhances traditional text search by leveraging semantic understanding. It allows users to find files and text snippets based on meaning and contextual similarity rather than exact keyword matches. It features an indexing process, a local vector database, and multiple output modes designed for various workflows, including RAG, code analysis, and general information retrieval from local files. This document details its architecture, features, UX considerations, and implementation strategy, focusing on robustness, usability, and performance.

---

**1. Overview**

Traditional `grep` is indispensable for pattern matching but falls short when searching for concepts or related information where exact phrasing is unknown or varies. `simgrep` addresses this by:

1.  **Indexing:** Processing local files, extracting text, chunking it, generating vector embeddings for each chunk using state-of-the-art models.
2.  **Storing:** Persisting these embeddings and associated metadata (file path, original text, offsets) in a local, efficient vector store (USearch) and metadata database (DuckDB).
3.  **Searching:** Taking a natural language query, embedding it, and retrieving the most semantically similar chunks from the indexed data.
4.  **Presenting:** Offering flexible output modes tailored to different user needs, from simple file lists to formatted contexts for Large Language Models (LLMs).

The tool is designed to be intuitive for simple use cases ("semantic grep replacement") while offering powerful configuration and features for advanced users.

**2. Goals**

*   **Accurate Semantic Search:** Provide relevant search results based on meaning, not just keywords.
*   **High Usability (Brilliant UX):**
    *   Simple, intuitive CLI commands with clear feedback.
    *   Works "out-of-the-box" for common scenarios with minimal setup.
    *   Sensible defaults with options for fine-tuning.
*   **Robustness & Reliability:**
    *   Graceful error handling.
    *   Data integrity for the index.
    *   Efficient handling of large file sets and updates.
*   **Performance:** Fast indexing (especially incremental) and search.
*   **Flexibility:** Support various file types and configurable embedding models.
*   **Versatile Output Modes:** Cater to diverse workflows including RAG, code retrieval, and information discovery.
*   **Import Graph Extraction:** Output complete dependency trees for supported programming languages (initially Python, TypeScript, Java).
*   **Lean & Modern Stack:** Utilize `uv` for project management, `mypy` for strict typing, DuckDB for metadata, and USearch for vector indexing.

**3. Non-Goals**

*   Real-time, OS-level file system monitoring for instant indexing (V1).
*   A graphical user interface (GUI) (V1).
*   Distributed indexing or search across multiple machines.
*   Managing or searching remote/cloud-based file storage directly.
*   Replacing version control systems or full-fledged document management systems.

**4. User Personas & Key Use Cases**

*   **Persona 1: The Developer ("DevDave")**
    *   **Needs:** Quickly find relevant code snippets, configuration examples, or documentation across multiple project repositories based on a problem description or concept. Copy code for use in IDE or with LLMs.
    *   **Use Cases:**
        *   `simgrep "how to implement async error handling in Python" ./backend_project --output copy-chunks --filetype .py`
        *   `simgrep "database connection pool settings" ./config_files/ --output show`
        *   `simgrep "user authentication flow diagram" ./docs/ --output paths`

*   **Persona 2: The Researcher/Analyst ("ResearcherRita")**
    *   **Needs:** Sift through large volumes of text documents (papers, articles, notes) to find information related to a hypothesis or research question. Prepare context for LLM-based summarization or question-answering.
    *   **Use Cases:**
        *   `simgrep "impact of climate change on Arctic ice melt" ~/research_papers/ --output rag --question "Summarize the key findings on Arctic ice melt due to climate change."`
        *   `simgrep "studies on transformer model attention mechanisms" ./llm_papers/ --output json > results.json`

*   **Persona 3: The Casual User ("CasualChris")**
    *   **Needs:** A simple way to find files or notes when they remember the gist but not the exact words or filename. A "smarter grep."
    *   **Use Cases:**
        *   `simgrep "that recipe for apple pie I saved" ~/documents/recipes/` (ephemeral search)
        *   `simgrep "meeting notes about the Q3 budget" ~/notes/ --output show`

**Intended Usage Summary:**
*   **Get answers about data (RAG):** Find relevant text, pass to LLM with a question.
*   **Copy related code/text:** Extract relevant chunks or full files for external use (e.g., with ChatGPT web UI).
*   **Show related content:** Display snippets and file paths.
*   **Grep replacement:** Simple, direct semantic search on a path: `simgrep "query" ./dir_or_file`.
*   **Detailed querying:** Index specific directories, filter by file type, tune parameters.
*   **"Out-of-the-box" & Tunable:** Works with minimal setup, but allows advanced configuration.

**5. Proposed Solution & Architecture**

**5.1. High-Level Architecture Diagram**

```
+-----------------------+      +-----------------------+      +-----------------------------+
|     CLI Interface     |----->|    Indexing Engine    |<---->|   Configuration Manager     |
| (Typer, Rich, Pydantic|      | (Strict Typing,       |      | (TOML, Pydantic Models,     |
| for input validation) |      |  Progress, Hashing)   |      |  Embedding Model Config)    |
+-----------------------+      +-----------------------+      +-----------------------------+
          |                           |        ^
          | (Query, Options, Path)    |        | (File Info, Hashes, Status)
          v                           v        |
+-----------------------+      +---------------------------------+
|    Search Engine      |      | File Processor & Embed. Generator|
| (Multiple Output Modes|      | (unstructured, sentence-trans., |
|  Clipboard Integration)|      |  Pydantic for Chunk Data)       |
+-----------------------+      +---------------------------------+
          |                           |
          | (Vector Query,            | (Vectors, Rich Metadata)
          |  Metadata Filters)        v
          v                      +-----------------------+
+-----------------------+        |      USearch Index    |
|      DuckDB           |<------>|  (Vector Storage)     |
| (Metadata: Projects,  |        +-----------------------+
|  Files, Chunks,       |
|  Offsets, Model Info) |
+-----------------------+
```

**5.2. Component Breakdown**

*   **CLI Interface (`cli.py`):**
    *   Technology: `Typer`, `Rich`.
    *   Responsibilities: Command parsing, argument validation (using Pydantic for complex inputs where beneficial), user interaction, formatted output, progress display.
*   **Configuration Manager (`config.py`):**
    *   Technology: TOML, `Pydantic`.
    *   Responsibilities: Manages global (`~/.config/simgrep/config.toml`) and potential project-specific (`.simgrep/config.toml`) configurations. Handles settings for DB paths, embedding models, default parameters, and API keys.
    *   Pydantic models define config structure for validation and typed access.
*   **Project Manager (`project.py`):**
    *   Responsibilities: Manages "projects" – distinct indexed sets of directories and their associated configurations. A default project is used if none is specified. Handles mapping of indexed paths to their DuckDB/USearch instances or namespaces.
*   **Indexing Engine (`indexer.py`):**
    *   Responsibilities: Orchestrates file discovery, calls File Processor, manages embedding generation, and stores data in DuckDB/USearch. Handles incremental updates based on file content hashes. Ensures data consistency.
*   **File Processor & Embedding Generator (`processor.py`):**
    *   Technology: `unstructured` (primary), `sentence-transformers`, `transformers` (for tokenizers).
    *   Responsibilities:
        *   File Parsing: Extracts text content from various file types.
        *   Text Chunking: Splits text into manageable, semantically coherent chunks using token-based strategies (configurable size and overlap).
        *   Embedding Generation: Converts text chunks into vector embeddings using the configured model.
*   **Import Dependency Analyzer (`dependency_analyzer.py`):**
    *   Technology: Python `ast`, TypeScript/Java parsers (`tree-sitter` or language-specific libraries).
    *   Responsibilities:
        *   Parse code files to detect `import` statements and resolve file/module paths.
        *   Recursively walk dependencies to build a full import tree for supported languages (Python, TypeScript, Java).
        *   Provide data structures for the Output Formatter to display or serialize the dependency graph.
*   **Vector Store (USearch) (`vector_store.py`):**
    *   Technology: `usearch`.
    *   Responsibilities: Stores, manages, and searches dense vector embeddings. Persists index to disk. Provides unique labels for vectors.
*   **Metadata Database (DuckDB) (`metadata_db.py`):**
    *   Technology: `duckdb`.
    *   Responsibilities: Stores metadata about indexed files, chunks (including original text, offsets, token counts), embedding model used, and project configurations. Facilitates filtering and joins.
*   **Search Engine (`searcher.py`):**
    *   Responsibilities: Takes a user query, generates its embedding, queries USearch for similar vectors, retrieves corresponding metadata from DuckDB, applies filters, and formats results for different output modes.
*   **Output Formatter (`formatter.py`):**
    *   Technology: `Rich`, `pyperclip`.
    *   Responsibilities: Handles presentation for various output modes (`show`, `paths`, `rag`, `copy-files`, `copy-chunks`, `json`).

**5.3. Data Models (Pydantic & DuckDB Schemas)**

*   **Pydantic Models (`models.py`):**
    *   `SimgrepConfig`: Global and project-level configuration.
        *   `db_directory`: Path to store DuckDB files and USearch indexes.
        *   `default_embedding_model`: Name of the default `sentence-transformers` model.
        *   `default_chunk_size`: Tokens per chunk.
        *   `default_chunk_overlap`: Token overlap.
        *   `llm_api_key`: Optional API key for RAG.
        *   `projects`: List of `ProjectConfig`.
    *   `ProjectConfig`:
        *   `name`: Unique project name.
        *   `indexed_paths`: List of directory/file paths.
        *   `file_extensions_include/exclude`: Optional lists.
        *   `embedding_model`: Specific model for this project (overrides default).
        *   (Other tunable params for this project)
    *   `ChunkData`: Internal representation for processed chunks.
        *   `text`: The chunk content.
        *   `source_file_id`: Reference to `indexed_files.file_id`.
        *   `usearch_label`: The label from USearch.
        *   `start_char_offset`, `end_char_offset`: Character offsets in the original file.
        *   `start_line`, `end_line`: Line numbers (best effort).
        *   `token_count`: Number of tokens in the chunk.

*   **DuckDB Schema (`metadata_db.py` defines this):**
    *   **Global DB (`~/.config/simgrep/global_metadata.duckdb`):**
        *   `projects`:
            *   `project_id BIGINT PRIMARY KEY`
            *   `project_name VARCHAR UNIQUE NOT NULL`
            *   `db_path VARCHAR NOT NULL`
            *   `usearch_index_path VARCHAR NOT NULL`
            *   `embedding_model_name VARCHAR NOT NULL`
        *   `project_indexed_paths`:
            *   `project_id BIGINT REFERENCES projects(project_id)`
            *   `path VARCHAR NOT NULL`
            *   `PRIMARY KEY (project_id, path)`
    *   **Project-specific DB (`.../projects/<name>/metadata.duckdb`):**
        *   `indexed_files`:
            *   `file_id BIGINT PRIMARY KEY`
            *   `file_path VARCHAR NOT NULL UNIQUE` (Absolute, resolved path)
            *   `content_hash VARCHAR NOT NULL` (SHA256 of file content)
            *   `file_size_bytes BIGINT`
            *   `last_modified_os TIMESTAMP`
            *   `last_indexed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP`
        *   `text_chunks`:
            *   `chunk_id BIGINT PRIMARY KEY`
            *   `file_id BIGINT REFERENCES indexed_files(file_id)`
            *   `usearch_label BIGINT UNIQUE NOT NULL` (Links to USearch vector)
            *   `chunk_text TEXT NOT NULL` (The full text of the chunk)
            *   `start_char_offset INTEGER NOT NULL`
            *   `end_char_offset INTEGER NOT NULL`
            *   `token_count INTEGER NOT NULL`
            *   `embedding_hash VARCHAR` (Hash of the embedding vector, for potential future use)
        *   `index_metadata`:
            *   `key VARCHAR PRIMARY KEY`
            *   `value VARCHAR NOT NULL`

**5.4. Configuration Management**

*   **Global Config:** `~/.config/simgrep/config.toml`. Stores defaults, global API keys. Created on first run or by `simgrep init --global`.
*   **Project-Based Config:**
    *   Projects are defined sets of indexed paths and their specific settings.
    *   A default project (`default_project`) is created in `db_directory` if no project is specified.
    *   Users can create named projects: `simgrep project create my_research_project`.
    *   Each project has its own DuckDB file and USearch index, ensuring separation.
*   **Overrides:** CLI options can override config settings for a single run.
*   **Tunable Parameters:** Chunk size, overlap, embedding model (per project), file inclusion/exclusion patterns.

**5.5. Embedding Model Management**

*   Models specified by their `sentence-transformers` names (e.g., `all-MiniLM-L6-v2`).
*   The model used for indexing is stored per project.
*   Changing a project's model will require a full re-index of that project's files. `simgrep index --rebuild` will be necessary.
*   The tool will download models on first use if not locally cached by `sentence-transformers`.

**6. Detailed Workflows**

**6.1. Initialization & Project Setup**

*   **First Run Experience:** If no global config exists, `simgrep` will offer to create one.
*   **`simgrep init`:**
    *   `simgrep init`: Initializes a new project in the current directory (creates `.simgrep/` with project config, local DB/index paths).
    *   `simgrep init --project-name <name> --path ./docs --path ./src`: Creates a named project with specified paths.
    *   `simgrep init --global`: Creates/overwrites the global config file.
*   **Implicit/Ephemeral Indexing (Grep-like usage):**
    *   `simgrep "query" ./some/path`
    1.  No project context is active for `./some/path`.
    2.  An ephemeral, in-memory DuckDB and USearch index are created.
    3.  `./some/path` is recursively scanned (if dir) or read (if file).
    4.  Files are processed, chunked, embedded, and added to the ephemeral index.
    5.  Search is performed against this temporary index.
    6.  Results are displayed.
    7.  Ephemeral index is discarded.
    *   This provides immediate utility without persistent state for one-off searches.
    *   A warning might suggest creating a project for faster subsequent searches on the same path.

**6.2. Indexing**

*   **`simgrep project add-path <path_to_index> [--project <name>]`:** Adds a path to the specified (or current/default) project's configuration and `indexed_paths` table.
*   **`simgrep index [--project <name>] [--force-recheck] [--rebuild]`:**
    1.  Load project configuration (or default).
    2.  Verify embedding model is available.
    3.  For each path in `project.indexed_paths`:
        *   Recursively scan for files, applying include/exclude filters.
        *   For each file:
            *   Calculate current content hash.
            *   Query DuckDB for existing entry (path, project_id).
            *   **Incremental Logic:**
                *   If new file: Process, embed, store (metadata in DuckDB, vector in USearch).
                *   If existing file & hash matches & not `--force-recheck`: Skip.
                *   If existing file & hash differs OR `--force-recheck`: Remove old chunks/vector from DuckDB/USearch, then re-process, embed, store.
                *   If `--rebuild`: Clear all data for the project and re-index everything.
    4.  Display `Rich` progress (files scanned, processed, time elapsed).
    5.  Save USearch index and commit DuckDB transaction.
    6.  Provide a summary: files added/updated/unchanged.
*   **Handling Deleted Files:** `simgrep index` can optionally scan for files in DB that no longer exist on disk and offer to prune them. `simgrep project clean --prune-deleted` for explicit action.

**6.3. Searching**

*   **`simgrep search "<query>" [path_or_project_target] [options]`:**
    1.  Determine target:
        *   If `path_or_project_target` is a file/directory path not in a project: Use ephemeral indexing (see 6.1).
        *   If `path_or_project_target` is a project name (`--project <name>` or inferred from current dir): Load that project's index.
        *   If no target, use default project or ask.
    2.  Load embedding model for the target project/ephemeral session.
    3.  Generate embedding for the `<query>`.
    4.  Query USearch for top-K most similar vector labels (K is configurable, default ~10-25).
    5.  Retrieve corresponding chunk metadata and file metadata from DuckDB using these labels.
    6.  Apply CLI filters (e.g., `--filetype .py`, `--similarity-threshold 0.7`).
    7.  Pass results to the Output Formatter based on `--output <mode>` (default: `show`).

*   **Output Modes (`--output <mode>`):**
    *   `show` (default): Displays `Rich.Table` with `File Path | Relevant Snippet (highlighted query terms if feasible) | Similarity Score | Chunk Location (line numbers)`.
    *   `paths`: Displays unique sorted list of file paths containing relevant chunks.
    *   `rag`:
        1.  Retrieves N most relevant chunks.
        2.  Formats them as context: "Context:\n[Chunk 1 Text]\nSource: [File Path 1, Chunk Location]\n\n[Chunk 2 Text]\nSource: [File Path 2, Chunk Location]..."
        3.  If `--question "<user_question>"` is provided: Prepends "Based on the following context, answer the question: <user_question>\n\n[Context]".
        4.  If LLM API key is configured: Sends the combined prompt to the LLM. Displays LLM response.
        5.  If no API key: Prints the fully formed prompt for the user to copy.
    *   `copy-files`: For each unique file containing relevant chunks, concatenates `--- File: <path> ---\n<full_file_content>\n\n` and copies to clipboard.
    *   `copy-chunks`: Concatenates only the relevant chunk texts (with their source file path as a comment/header) and copies to clipboard.
    *   `json`: Outputs detailed results (file path, chunk text, score, offsets, metadata) as a JSON array.
    *   `count`: Shows number of matching chunks and files.
    *   `imports`: Displays the full dependency tree for a single code file, recursively resolving imports for Python, TypeScript, and Java.

**7. User Experience (UX) Design**

**7.1. Command-Line Interface (CLI)**

*   **Structure:**
    *   `simgrep <command> [subcommand] [arguments] [options]`
    *   Top-level commands: `search`, `index`, `init`, `config`, `project`, `status`, `clean`.
    *   `project` subcommands: `create`, `list`, `add-path`, `remove-path`, `set-param`.
*   **Clarity:** Consistent naming. Comprehensive help (`-h`, `--help`) via `Typer`.
*   **Defaults:** Sensible defaults for chunk size, overlap, model, number of results.
*   **Simplicity for common tasks:** `simgrep "query" ./path` is the hero command for quick, ephemeral searches.

**7.2. Feedback and Interactivity**

*   **Progress:** `Rich` progress bars for indexing and other long operations.
*   **Status Messages:** Clear, concise messages for success, warnings, errors.
*   **`simgrep status [--project <name>]`:** Shows indexed paths, number of files/chunks, DB size, embedding model, last index time for the project.
*   **Confirmation:** Prompts for destructive actions (e.g., `index --rebuild`, `project delete`). Use `--yes` to skip prompts in automation.
*   **Error Reporting:** User-friendly error messages with suggestions if possible. Stack traces hidden by default, shown with `--verbose`.

**7.3. Defaults and Tunability**

*   **Out-of-the-box:** Works immediately for ephemeral search. Default project for persistent indexing requires minimal setup.
*   **Tunability:**
    *   Via `config.toml` (global or project-specific).
    *   Via CLI options overriding config for a single run.
    *   Parameters: embedding model, chunk size/overlap, file filters, similarity thresholds, LLM API endpoint/model.

**8. Robustness and Reliability**

**8.1. Error Handling**

*   Strict typing with `mypy` (strictest config) to catch errors early.
*   Pydantic for input and configuration validation.
*   Graceful handling of file I/O errors (skip unreadable files/paths with warnings).
*   Robust parsing of diverse file types (leverage `unstructured`'s capabilities and error handling).
*   DB connection management and error handling.
*   Network error handling for LLM API calls.

**8.2. Data Integrity**

*   DuckDB transactions for metadata updates to ensure consistency.
*   Content hashing (SHA256) for reliable detection of file changes.
*   USearch index persistence (save/load).
*   Regularly test index backup and restore if this becomes a feature.

**8.3. Performance Considerations**

*   **Indexing:**
    *   Batch embedding generation.
    *   Batch writes to DuckDB/USearch.
    *   Efficient file traversal and hash computation.
    *   Parallel processing for embedding generation (CPU-bound) if feasible using `multiprocessing` or `asyncio` with process pool executors.
*   **Search:**
    *   USearch is highly optimized for k-NN search.
    *   Efficient metadata retrieval from DuckDB (indexed columns).
*   **Memory Usage:** Monitor memory footprint, especially with large models or many files. `sentence-transformers` can be memory-intensive.

**9. Future Considerations (Post V1)**

*   **Daemon Mode:** Background process for continuous, real-time indexing of monitored paths.
*   **Advanced RAG:** Citation of sources within LLM response, re-ranking of chunks by LLM.
*   **UI/Web Interface:** A simple web interface for searching and visualizing results.
*   **More Sophisticated Chunking:** Language-aware chunking, code-specific chunking (functions/classes as primary units).
*   **Index Migration Tool:** For handling schema changes or USearch version upgrades.
*   **Support for other embedding providers:** OpenAI API, Cohere, etc.
*   **Pre-computed Project Indexes:** Ability to share pre-computed indexes for common codebases/document sets.

**10. Technical Stack Summary**

*   **CLI Framework:** `Typer`, `Rich`
*   **Configuration:** TOML, `Pydantic`
*   **Vector Indexing:** `usearch`
*   **Metadata Storage:** `duckdb`
*   **Embeddings:** `sentence-transformers`, `transformers` (for tokenizers)
*   **File Parsing:** `unstructured`
*   **Build/Package Management:** `uv`
*   **Linting/Formatting:** `Ruff`
*   **Type Checking:** `mypy` (strictest)
*   **Clipboard:** `pyperclip`