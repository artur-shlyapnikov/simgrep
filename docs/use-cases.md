# Use Cases

`simgrep` supports several workflows for searching and answering questions over your local files.

## Ephemeral Search

For ad-hoc queries, `simgrep` builds an index in `~/.config/simgrep/ephemeral_cache/<hash>`.
If that cache already exists it is reused for faster results.

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant simgrep
    participant USearch
    participant DuckDB
    User->>simgrep: simgrep search "query" path
    simgrep->>DuckDB: store chunks in cache
    simgrep->>USearch: build or load cached index
    simgrep->>USearch: query embeddings
    USearch-->>simgrep: similar chunks
    simgrep->>User: display results
```

## Persistent Indexing & Search

Projects can be indexed once and searched many times. Index data is saved on disk.

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant simgrep
    participant USearch
    participant DuckDB
    User->>simgrep: simgrep index project/
    simgrep->>USearch: add embeddings to disk
    simgrep->>DuckDB: record metadata
    User->>simgrep: simgrep search "query"
    simgrep->>USearch: lookup embeddings
    USearch-->>simgrep: results
    simgrep->>User: show matches
```

## RAG-Based Answering

`simgrep` can retrieve relevant chunks and feed them to a language model to answer questions.

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant simgrep
    participant USearch
    participant DuckDB
    User->>simgrep: simgrep answer "question"
    simgrep->>USearch: retrieve top chunks
    USearch-->>simgrep: return passages
    simgrep->>User: final answer (via LLM)
```
