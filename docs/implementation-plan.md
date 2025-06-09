**Implementation Plan: `simgrep`**

**Phase 1: Ephemeral Single Text File Search (First E2E Slice)**

*Goal: Implement `simgrep "query" <single_text_file_path>` with in-memory processing and basic "show" output.*

* **Deliverable 1.1: Text Extraction from Single File** ✅
  * **Goal:** Read content from a specified text file.
  * **Tasks:**
        1. `uv add unstructured`
        2. In `simgrep/processor.py`, create `extract_text_from_file(file_path: Path) -> str` using `unstructured.partition.auto.partition_text` or similar for `.txt` files. Handle file not found.
        3. Integrate into `search` command in `main.py` to call this.
  * **Key Modules:** `simgrep/processor.py`, `simgrep/main.py`
  * **What to Test:**
    * Create `test.txt` with "Hello world. This is a test."
    * `simgrep search "anything" ./test.txt` should internally have the content of `test.txt` (e.g., print it for now).
    * Test with a non-existent file path – should show a user-friendly error.

* **Deliverable 1.2: Simplistic Text Chunking** ✅
  * **Goal:** Break extracted text into smaller, overlapping pieces.
  * **Tasks:**
        1. In `simgrep/processor.py`, create `chunk_text_simple(text: str, chunk_size_chars: int, overlap_chars: int) -> List[str]`. (Focus on simplicity, e.g., character-based, not token-based yet).
        2. Integrate into `search` command to chunk the extracted text.
  * **Key Modules:** `simgrep/processor.py`, `simgrep/main.py`
  * **What to Test:**
    * Using `test.txt` from 1.1, `simgrep search "anything" ./test.txt` should internally produce a list of text chunks (e.g., print them). Verify chunking and overlap logic manually.

* **Deliverable 1.3: Embedding Generation (Query & Chunks)** ✅
  * **Goal:** Convert query and text chunks into vector embeddings.
  * **Tasks:**
        1. `uv add sentence-transformers` (implicitly `transformers`, `torch`).
        2. In `simgrep/processor.py`, create `generate_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray`. This function will download the model on first use.
        3. Integrate into `search` command to embed the query and the generated chunks.
  * **Key Modules:** `simgrep/processor.py`, `simgrep/main.py`
  * **What to Test:**
    * `simgrep search "world" ./test.txt` should internally generate embeddings for "world" and for each chunk of `test.txt`. Print shapes of embedding arrays to verify. (Model download might take time on first run).

* **Deliverable 1.4: In-Memory Vector Search (USearch)** ✅
  * **Goal:** Find most similar chunks to the query using an in-memory vector index.
  * **Tasks:**
        1. `uv add usearch numpy` (numpy might come with sentence-transformers).
        2. In `simgrep/vector_store.py`:
            *`create_inmemory_index(embeddings: np.ndarray) -> usearch.index.Index`.
            * `search_inmemory_index(index: usearch.index.Index, query_embedding: np.ndarray, k: int) -> List[Tuple[int, float]]` (returns (chunk_original_index, distance/similarity)).
        3. Integrate into `search` command.
  * **Key Modules:** `simgrep/vector_store.py`, `simgrep/main.py`
  * **What to Test:**
    * `simgrep search "world" ./test.txt` should internally get a list of (chunk_index, score) tuples. Print these to verify USearch is working.

* **Deliverable 1.5: Basic "Show" Output (E2E!)** ✅
  * **Goal:** Display the most relevant chunk text and its source file.
  * **Tasks:**
        1. In `simgrep/formatter.py`, create `format_show_basic(file_path: Path, chunk_text: str, score: float) -> str`.
        2. Modify `search` command in `main.py`:
            *After getting results from USearch, retrieve the original text of the top matching chunk.
            * Use `formatter.py` to print the file path, the chunk text, and the similarity score.
  * **Key Modules:** `simgrep/formatter.py`, `simgrep/main.py`
  * **What to Test (E2E):**
    * `simgrep search "world" ./test.txt` should print something like:

            ```
            File: test.txt
            Score: 0.85
            Chunk: Hello world. This is a test.
            ```

    * Test with a query that *doesn't* match well to see low scores or no output (if thresholding, though not implemented yet).

---

**Phase 2: Enhancing Ephemeral Search & Core Data Structures**

*Goal: Expand ephemeral search to directories, improve chunking, introduce Pydantic models, and use in-memory DuckDB for metadata.*

* **Deliverable 2.1: Directory Traversal & Multiple File Processing** ✅
  * **Goal:** Allow `path_to_search` to be a directory, processing all `.txt` files within.
  * **Tasks:**
        1. Modify `search` command logic: if `path_to_search` is a directory, use `Path.rglob("*.txt")` to find files.
        2. Process each file (extract, chunk, embed) and aggregate all chunks and their source file info.
        3. Update USearch indexing to handle chunks from multiple files (ensure labels/IDs map back correctly).
  * **Key Modules:** `simgrep/main.py`, `simgrep/processor.py`
  * **What to Test (E2E):**
    * Create `dir1/a.txt` ("apple banana") and `dir1/b.txt` ("banana orange").
    * `simgrep search "banana" ./dir1` should show results from both `a.txt` and `b.txt`.

* **Deliverable 2.2: Pydantic Models for Data & Token-based Chunking** ✅
  * **Goal:** Introduce structured data handling and more robust chunking.
  * **Tasks:**
        1. In `simgrep/models.py`, define `ChunkData(text: str, source_file_path: Path, source_file_id: int, usearch_label: int, start_char_offset: int, end_char_offset: int, token_count: int)`.
        2. In `simgrep/processor.py`:
            *Use `sentence-transformers` tokenizer (e.g., `AutoTokenizer.from_pretrained("all-MiniLM-L6-v2")`).
            * Update `chunk_text` to use token counts for `chunk_size`, `overlap`. Calculate character offsets and token counts for each `ChunkData`.
        3. Refactor `search` to use `List[ChunkData]`.
  * **Key Modules:** `simgrep/models.py`, `simgrep/processor.py`, `simgrep/main.py`
  * **What to Test:**
    * Internal representation of chunks now includes offsets and token counts. Print these for verification.
    * Chunking should be more semantically aware (though still basic).

* **Deliverable 2.3: In-Memory DuckDB for Ephemeral Metadata** ✅
  * **Goal:** Use DuckDB to manage chunk metadata during ephemeral search.
  * **Tasks:**
        1. `uv add duckdb`
        2. In `simgrep/metadata_db.py`:
            *`create_inmemory_db_connection() -> duckdb.DuckDBPyConnection`.
            * Define schema and create functions for `temp_files (file_id PK, file_path TEXT)` and `temp_chunks (chunk_id PK, file_id FK, usearch_label INT, text_snippet TEXT, start_offset INT, end_offset INT)`.
            * Functions to insert `ChunkData` into these tables. USearch will store `chunk_id` as its label.
        3. Modify `search` command: Populate in-memory DuckDB. After USearch returns `chunk_id`s, query DuckDB to get full `ChunkData`.
  * **Key Modules:** `simgrep/metadata_db.py`, `simgrep/main.py`, `simgrep/models.py`
  * **What to Test:**
    * Search results should still be correct. Internally, data is now flowing through DuckDB.
    * (Optional) Add a debug flag to dump DuckDB table contents.

* **Deliverable 2.4: `--output paths` Mode** ✅
  * **Goal:** Add an option to list unique file paths containing matches.
  * **Tasks:**
        1. In `simgrep/formatter.py`, add `format_paths(chunk_results: List[ChunkData]) -> str`.
        2. In `simgrep/main.py`, add `--output` option (Enum: "show", "paths") to `search`.
        3. Call the appropriate formatter.
  * **Key Modules:** `simgrep/formatter.py`, `simgrep/main.py`
  * **What to Test (E2E):**
    * `simgrep search "banana" ./dir1 --output paths` should print:

            ```
            ./dir1/a.txt
            ./dir1/b.txt
            ```

**Phase 3: Persistent Default Project Indexing**

*Goal: Implement `simgrep index <path>` and `simgrep search <query>` (without path) using a persistent default project index.*

* **Deliverable 3.1: Basic Global Configuration (`config.py`)** ✅
  * **Goal:** Define where persistent data for the default project will live.
  * **Tasks:**
        1. In `simgrep/models.py`, define `SimgrepConfig(db_directory: Path = Path("~/.config/simgrep/default_project").expanduser(), ...)`
        2. In `simgrep/config.py`, function `load_or_create_global_config() -> SimgrepConfig`. For now, it just returns a default `SimgrepConfig` object. Create the `db_directory` if it doesn't exist.
        3. (No TOML file yet, just programmatic defaults for simplicity).
  * **Key Modules:** `simgrep/models.py`, `simgrep/config.py`
  * **What to Test:**
    * Running any `simgrep` command creates `~/.config/simgrep/default_project` if it doesn't exist.

* **Deliverable 3.2: Persistent DuckDB & USearch for Default Project** ✅
  * **Goal:** Store index data on disk.
  * **Tasks:**
        1. Modify `simgrep/metadata_db.py`:
            *`connect_persistent_db(db_path: Path) -> duckdb.DuckDBPyConnection`.
            * Define full schemas for `indexed_files` and `text_chunks` (as per design doc) and create tables if they don't exist.
        2. Modify `simgrep/vector_store.py`:
            *`load_persistent_index(index_path: Path) -> usearch.index.Index`.
            * `save_persistent_index(index: usearch.index.Index, index_path: Path)`.
  * **Key Modules:** `simgrep/metadata_db.py`, `simgrep/vector_store.py`, `simgrep/config.py`
  * **What to Test:**
    * Functions can be called. Test save/load of an empty/trivial index and DB.

* **Deliverable 3.3: `simgrep index <path>` Command (Non-Incremental)** ✅
  * **Goal:** Create a persistent index for a given path under the default project.
  * **Tasks:**
        1. Add `index` command to `simgrep/main.py`: `simgrep index <path_to_index>`.
        2. In `simgrep/indexer.py`:
            *Orchestrate: load global config for paths. Connect to persistent DB/load USearch index (or create if new).
            * Scan `<path_to_index>` (like ephemeral directory scan).
            *For each file: process (extract, chunk using token-based chunker, embed).
            * Store `IndexedFile` metadata and `TextChunk` metadata in persistent DuckDB.
            *Add embeddings to USearch index.
            * Save USearch index and commit DuckDB.
            * *Crucially, for now, this command will WIPE existing default project data and re-index from scratch.*
        3. Use `Rich` progress bar.
  * **Key Modules:** `simgrep/main.py`, `simgrep/indexer.py`, `simgrep/processor.py`, `simgrep/metadata_db.py`, `simgrep/vector_store.py`, `simgrep/config.py`
  * **What to Test:**
    * `simgrep index ./dir1`. Check that `~/.config/simgrep/default_project/metadata.duckdb` and `index.usearch` are created/populated.
    * (Manual) Inspect DB tables to verify content.

* **Deliverable 3.4: `simgrep search <query>` (Using Default Persistent Index)** ✅
  * **Goal:** Search against the previously built persistent index.
  * **Tasks:**
        1. Modify `search` command in `main.py`:
            *If `path_to_search` argument is *omitted*, load default project's persistent DB/USearch index.
            * Embed query, search USearch, retrieve metadata from DuckDB, format.
            * If `path_to_search` *is* provided, it still performs ephemeral search (as in Phase 1-2).
  * **Key Modules:** `simgrep/main.py`, `simgrep/searcher.py` (refactor search logic here), `simgrep/metadata_db.py`, `simgrep/vector_store.py`
  * **What to Test (E2E):**
        1. `simgrep index ./dir1`
        2. `simgrep search "banana"` (no path) should show results from `./dir1` based on the persistent index.
        3. `simgrep search "apple" ./another_dir/some.txt` should still do ephemeral search on `some.txt`.

---

* **Deliverable 3.5: Basic `simgrep status` Command** ✅ **(Done)**
  * **Goal:** Show basic information about the default project's index.
  * **Tasks:**
        1. Add `status` command to `main.py`.
        2. It should connect to the default project's DuckDB and query `indexed_files` and `text_chunks` for counts.
        3. Print: "Default Project: X files indexed, Y chunks."
  * **Key Modules:** `simgrep/main.py`, `simgrep/metadata_db.py`
  * **What to Test:**
    * `simgrep index ./dir1`
    * `simgrep status` should show correct counts.

---

**Phase 4: Incremental Indexing & Basic Project Management**

*Goal: Make indexing efficient by only processing changed files and introduce the concept of named projects.*

* **Deliverable 4.1: Content Hashing & Storing in DB** ✅
  * **Goal:** Calculate and store file content hashes to detect changes.
  * **Tasks:**
        1. In `simgrep/processor.py`, add `calculate_file_hash(file_path: Path) -> str` (SHA256).
        2. Modify `indexed_files` table schema in `metadata_db.py` to include `content_hash VARCHAR`, `last_modified_os TIMESTAMP`.
        3. Update `indexer.py` (`simgrep index`) to calculate and store these for each file.
  * **Key Modules:** `simgrep/processor.py`, `simgrep/metadata_db.py`, `simgrep/indexer.py`
  * **What to Test:**
    * After `simgrep index ./dir1`, inspect `indexed_files` table to see hashes and mtimes.

* **Deliverable 4.2: Basic Incremental Indexing Logic** ✅
  * **Goal:** Skip unchanged files and re-index changed/new files.
  * **Tasks:**
        1. Modify `indexer.py` (`simgrep index`):
            *For each file found on disk:
                * Check if it's in `indexed_files` for the project.
                *If new: process and add.
                * If existing: compare current hash & mtime with stored values.
                    *If same: skip (log "unchanged").
                    * If different:
                        *Remove old chunks for this file from `text_chunks` and USearch.
                        * Re-process file, add new chunks to DB and USearch. Update `indexed_files` entry.
            * (Defer pruning of files deleted from disk for now).
  * **Key Modules:** `simgrep/indexer.py`, `simgrep/metadata_db.py`, `simgrep/vector_store.py`
  * **What to Test:**
        1. `simgrep index ./dir1`.
        2. Run `simgrep index ./dir1` again. Observe faster execution and "unchanged" logs.
        3. Modify a file in `dir1`. Run `simgrep index ./dir1`. Observe it re-indexes only that file.
        4. Add a new file to `dir1`. Run `simgrep index ./dir1`. Observe it indexes the new file.

* **Deliverable 4.3: `index --rebuild` Option** ✅
  * **Goal:** Allow forcing a full re-index of the default project.
  * **Tasks:**
        1. Add `--rebuild` flag to `simgrep index` command.
        2. If present, `indexer.py` should delete all records from `text_chunks` and `indexed_files` for the project, delete the USearch file, then proceed as if indexing from scratch.
  * **Key Modules:** `simgrep/main.py`, `simgrep/indexer.py`
  * **What to Test:**
    * `simgrep index ./dir1 --rebuild` clears and re-indexes everything for the default project.

* **Deliverable 4.4: TOML Configuration File & Named Projects Foundation** ✅
  * **Goal:** Introduce `config.toml` and schema for managing multiple projects.
  * **Tasks:**
        1. Modify `simgrep/config.py`:
            *`load_or_create_global_config()` now reads/writes `~/.config/simgrep/config.toml`.
            * `SimgrepConfig` Pydantic model updated to reflect TOML structure (e.g., `db_directory`, `default_embedding_model`, `projects: Dict[str, ProjectConfig]`).
            *`ProjectConfig` Pydantic model (`name`, `indexed_paths: List[Path]`, `embedding_model`, etc.).
        2. Update `metadata_db.py` (for a *global* DB, e.g., `~/.config/simgrep/simgrep_meta.duckdb`):
            * Add `projects` table (`project_id PK`, `project_name UNIQUE`, `db_path`, `usearch_index_path`, `embedding_model_name`, etc.).
            * Add `project_indexed_paths` table (`path_id PK`, `project_id FK`, `absolute_path UNIQUE`).
        3. On first run or if `config.toml` is minimal, create a "default" project entry in the global DB and `config.toml`.
  * **Key Modules:** `simgrep/config.py`, `simgrep/models.py`, `simgrep/metadata_db.py`
  * **What to Test:**
    * `~/.config/simgrep/config.toml` is created/updated.
    * A global DuckDB file is created with `projects` table. "default" project exists.

* **Deliverable 4.5: `simgrep project` Subcommands (create, list)** ✅
  * **Goal:** Allow users to create and list named projects.
  * **Tasks:**
        1. In `main.py`, define a new Typer command group `project`.
        2. Implement `project create <name>`:
            *Validate that `<name>` is not already present in the global `projects` table.
            * Determine the project's data directory: `~/.config/simgrep/projects/<name>/`.
            *Create this directory and paths for `metadata.duckdb` and `index.usearch`.
            * Insert a row into the global `projects` table with these paths and the default embedding model.
            *Add a matching entry to `config.toml` under `projects` (use helper in `config.py`).
        3. Implement `project list`:
            * Query `projects` table via `metadata_db.get_all_projects()` (new helper) and print project names.
            * Mark the default project in the output for clarity.
        4. Expose these subcommands under `simgrep project`.
  * **Key Modules:** `simgrep/main.py`, `simgrep/config.py`, `simgrep/metadata_db.py`
  * **What to Test:**
    * `simgrep project create my_research` creates `~/.config/simgrep/projects/my_research/` with DB and index files.
    * `simgrep project list` shows "default" and "my_research" (default flagged).
    * `config.toml` contains a `[projects.my_research]` section matching the created paths.

* **Deliverable 4.6: `index` & `search` with `--project <name>`** ✅
  * **Goal:** Target indexing and searching to specific named projects.
  * **Tasks:**
        1. Add `--project <name>` option to `simgrep index` and `simgrep search`.
        2. Modify `indexer.py` and `searcher.py` (or `main.py` logic) to use the specified project's DB/USearch paths and configuration. Default to "default" project if option omitted.
  * **Key Modules:** `simgrep/main.py`, `simgrep/indexer.py`, `simgrep/searcher.py`, `simgrep/config.py`
  * **What to Test:**
        1. `simgrep project create projA`
        2. `simgrep index ./dirA --project projA`
        3. `simgrep search "termA" --project projA` (uses projA's index).
        4. `simgrep search "termA"` (uses default project's index, likely no results for termA).

* **Deliverable 4.7: `simgrep project add-path <path> [--project <name>]`** ✅
  * **Goal:** Associate multiple directory/file paths with a project for indexing.
  * **Tasks:**
        1. Add `project add-path <path_to_add>` subcommand.
        2. Stores path in `project_indexed_paths` table for the specified (or default) project.
        3. `simgrep index [--project <name>]` now iterates over all paths registered for that project in `project_indexed_paths`.
  * **Key Modules:** `simgrep/main.py`, `simgrep/indexer.py`, `simgrep/config.py`, `simgrep/metadata_db.py`
  * **What to Test:**
        1. `simgrep project create myproj`
        2. `simgrep project add-path ./docs --project myproj`
        3. `simgrep project add-path ./src --project myproj`
        4. `simgrep index --project myproj` indexes both `./docs` and `./src`.

---

**Phase 5: Output Modes & UX Refinements**

*Goal: Implement all specified output modes and improve CLI usability.*

* **Deliverable 5.1: `json` Output Mode** ✅
  * **Goal:** Provide detailed structured output.
  * **Tasks:**
        1. In `formatter.py`, implement `format_json(results: List[ChunkData]) -> str`.
        2. Add "json" to `--output` Enum in `main.py`.
  * **Key Modules:** `simgrep/formatter.py`, `simgrep/main.py`, `simgrep/models.py`
  * **What to Test (E2E):** `simgrep search "test" --output json > results.json`. Validate JSON.

* **Deliverable 5.2: `copy-chunks` and `copy-files` Output Modes**
  * **Goal:** Easy copying of results to clipboard.
  * **Tasks:**
        1. `uv add pyperclip`
        2. In `formatter.py`, implement `format_copy_chunks(results: List[ChunkData])` and `format_copy_files(results: List[ChunkData])`. These will use `pyperclip.copy()`.
        3. Add to `--output` Enum.
  * **Key Modules:** `simgrep/formatter.py`, `simgrep/main.py`
  * **What to Test (E2E):** Verify clipboard content after running these commands.

* **Deliverable 5.3: Basic `rag` Output Mode (Context Formatting Only)**
  * **Goal:** Prepare context for an LLM, print to stdout.
  * **Tasks:**
        1. In `formatter.py`, implement `format_rag_context(results: List[ChunkData], question: Optional[str]) -> str`.
        2. Add "rag" to `--output` Enum. Add optional `--question <text>` to `search` command.
  * **Key Modules:** `simgrep/formatter.py`, `simgrep/main.py`
  * **What to Test (E2E):** `simgrep search "info" --output rag --question "Summarize?"` prints formatted prompt.

* **Deliverable 5.4: `count` Output Mode**
  * **Goal:** Show counts of matching chunks and files.
  * **Tasks:**
        1. In `formatter.py`, implement `format_count(results: List[ChunkData]) -> str`.
        2. Add "count" to `--output` Enum.
  * **Key Modules:** `simgrep/formatter.py`, `simgrep/main.py`
  * **What to Test (E2E):** `simgrep search "test" --output count` shows "X matching chunks in Y files."

* **Deliverable 5.5: Enhanced `show` Output with `Rich.Table`**
  * **Goal:** Improve readability of default output.
  * **Tasks:**
        1. Refactor `format_show_basic` in `formatter.py` to use `Rich.Table`.
        2. Include columns: File Path, Snippet (try to highlight query terms simply), Score, Location (line numbers if feasible - requires line number tracking in `ChunkData` which is a stretch goal for V1, char offsets are fine).
  * **Key Modules:** `simgrep/formatter.py`
  * **What to Test (E2E):** `simgrep search "test"` output is a well-formatted table.

* **Deliverable 5.6: File Type Filtering (Include/Exclude in Project Config)**
  * **Goal:** Allow users to specify which file types to index.
  * **Tasks:**
        1. Add `file_extensions_include: List[str]` and `file_extensions_exclude: List[str]` to `ProjectConfig` Pydantic model.
        2. Update `simgrep project create/set-param` to manage these.
        3. Modify `indexer.py` file scanning logic to respect these filters.
  * **Key Modules:** `simgrep/models.py`, `simgrep/config.py`, `simgrep/indexer.py`, `simgrep/main.py` (for project set-param)
  * **What to Test:** Create a project, set include `.py`. Index a dir with `.py` and `.md` files. Only `.py` files should be indexed (verify with `status` or search).

---

**Phase 6: Advanced Features & Robustness**

*Goal: Implement RAG with LLM, project-local config, strict typing, and error handling polish.*

* **Deliverable 6.1: `simgrep init` Command**
  * **Goal:** Easy setup of global config and project-local config.
  * **Tasks:**
        1. Add `init` command to `main.py`.
        2. `simgrep init --global`: Creates/overwrites `~/.config/simgrep/config.toml` and global DB.
        3. `simgrep init`: If in a directory not already part of a known project, offers to create a `.simgrep/config.toml` for a new project rooted there.
  * **Key Modules:** `simgrep/main.py`, `simgrep/config.py`
  * **What to Test:**
    * `simgrep init --global` creates the global config.
    * `cd my_new_project_dir; simgrep init` creates `.simgrep/config.toml`.

* **Deliverable 6.2: Project-Specific Configuration Overrides (`.simgrep/config.toml`)**
  * **Goal:** Allow per-project settings to override global defaults.
  * **Tasks:**
        1. Modify `config.py` to load `.simgrep/config.toml` if present in a project's root and merge its settings over global/default project settings.
        2. The `indexer` and `searcher` should use this resolved config.
  * **Key Modules:** `simgrep/config.py`, `simgrep/indexer.py`, `simgrep/searcher.py`
  * **What to Test:** Set a different `embedding_model` in `.simgrep/config.toml`. `simgrep index` for that project should use it (verify via logs or stored project metadata).

* **Deliverable 6.3: Configurable Embedding Model per Project**
  * **Goal:** Allow changing embedding model and trigger re-index.
  * **Tasks:**
        1. Ensure `embedding_model_name` is stored in `projects` table (global DB) and `ProjectConfig`.
        2. `simgrep project set-param embedding_model <model_name> --project <name>`.
        3. Warn user that re-index (`index --rebuild`) is necessary.
  * **Key Modules:** `simgrep/main.py`, `simgrep/config.py`, `simgrep/models.py`, `simgrep/metadata_db.py`
  * **What to Test:** Change model, re-index, search. Observe different model being loaded.

* **Deliverable 6.4: `rag` Output Mode with LLM Integration (Optional API Call)**
  * **Goal:** If API key is configured, send context to LLM and display response.
  * **Tasks:**
        1. Add `llm_api_key` (optional) to `SimgrepConfig` (global or project).
        2. `uv add litellm` (or `openai`).
        3. Modify `format_rag_context` or a new function in `formatter.py`: if API key and `--question` are present, make API call.
  * **Key Modules:** `simgrep/formatter.py`, `simgrep/config.py`, `simgrep/models.py`
  * **What to Test (E2E if API key available):** `simgrep search "info" --output rag --question "Summarize?"` (with API key in config) prints LLM response. If no key, prints prompt.

* **Deliverable 6.5: `mypy --strict` Pass & Error Handling Polish**
  * **Goal:** Improve code quality and robustness.
  * **Tasks:**
        1. Configure `mypy` for `--strict` in `pyproject.toml`.
        2. Add type hints and resolve all `mypy` errors.
        3. Review code for error handling:
            *File I/O errors during indexing (skip unreadable files with warnings).
            * Graceful handling of empty search results.
            * User-friendly messages for config errors.
        4. Confirmation prompts for destructive actions (e.g., `index --rebuild`, `project delete` if added).
  * **Key Modules:** Entire codebase.
  * **What to Test:**
    * `uv run mypy simgrep/` passes under strict.
    * Manually test various error conditions (bad paths, unreadable files, bad config).

---

**Phase 7: Import Dependency Extraction**

*Goal: Provide a way to output the full import tree for a given source file.*

* **Deliverable 7.1: `dependency_analyzer.py`**
  * **Goal:** Parse direct imports for supported languages.
  * **Tasks:**
        1. Create new module `dependency_analyzer.py` with language-specific parsers (Python `ast`, TypeScript/Java via `tree-sitter` or similar).
        2. Implement `get_direct_imports(file_path: Path) -> List[Path]` resolving files within indexed paths.
  * **Key Modules:** `simgrep/dependency_analyzer.py`
  * **What to Test:** Unit tests on sample Python/TS/Java files verifying import detection.

* **Deliverable 7.2: Recursive Import Graph**
  * **Goal:** Build a dependency tree by following imports recursively.
  * **Tasks:**
        1. Add function `build_import_tree(file_path: Path, seen: Set[Path]) -> Dict[Path, List[Path]]`.
        2. Handle cycles gracefully and limit traversal to project files.
  * **Key Modules:** `dependency_analyzer.py`
  * **What to Test:** Import graph generation on small code samples with nested imports.

* **Deliverable 7.3: `imports` Output Mode**
  * **Goal:** Expose import tree via the CLI.
  * **Tasks:**
        1. Add `imports` to `--output` Enum in `main.py` and `formatter.py`.
        2. Format results as a readable tree or JSON list of paths.
  * **Key Modules:** `simgrep/main.py`, `simgrep/formatter.py`, `simgrep/dependency_analyzer.py`
  * **What to Test (E2E):** `simgrep search main.py --output imports` shows the transitive list of imports.

---

**Cross-Cutting Concerns (Throughout Development):**

* **Version Control:** `git init` from D0.1. Commit after each deliverable.
* **Testing:**
  * **Unit Tests (`tests/unit/`):** For pure functions in `processor.py`, `formatter.py`, config loading logic. Start adding these from Phase 2.
  * **Integration Tests (`tests/integration/`):** For interactions between components (e.g., `indexer` correctly writing to `metadata_db` and `vector_store`). Start from Phase 3.
  * **E2E Tests (`tests/e2e/`):** Shell scripts or Python scripts using `subprocess` to run CLI commands and assert stdout/stderr/file outputs. Continuously expand.
* **Documentation:**
  * Update `README.md` with installation, basic usage as features are added.
  * Ensure `simgrep <command> --help` is always informative.
* **Linting/Formatting:** Run `uv run ruff format .` and `uv run ruff check --fix .` regularly.
