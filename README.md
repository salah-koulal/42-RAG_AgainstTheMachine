# RAG against the machine

A local Retrieval-Augmented Generation (RAG) system built over the vLLM (0.10.1) codebase and documentation. It indexes source files using **BM25**, retrieves relevant code spans with exact character offsets, and generates grounded answers using **Qwen/Qwen3-0.6B**.

---

## 1. System Architecture

### Pipeline Flow

```text
[Raw Files: .py, .md, .txt]
          │
          ▼
[Format-Specific Chunking] ──► (Track file_path + start/end character offsets)
          │
          ▼
[BM25 Indexing (bm25s)] ───► Saved to data/processed/ (Index + Metadata)
          │
[Query] ──┴─► [BM25 Retrieval] ──► [Top-k MinimalSource Spans]
                                             │
                                             ▼
                                 [Extract Text Slices from Files]
                                             │
                                             ▼
[Qwen3-0.6B Prompt Augmented with Context] ─► [Generated Answer]
```

### Components and Interactions

| Component | Module | Responsibility |
|---|---|---|
| **Chunker** | `src/chunking/` | Splits `.py` by syntax boundaries and `.md`/`.txt` recursively, tracking character offsets. |
| **Indexer** | `src/indexing/` | Builds sparse BM25 matrix, saves metadata, and manages incremental updates. |
| **Retriever** | `src/retrieval/` | Scores and ranks chunks using BM25, returning top-k `MinimalSource` objects. |
| **Generator** | `src/generation/` | Slices context from files and prompts `Qwen/Qwen3-0.6B` to generate grounded answers. |
| **CLI** | `src/cli.py` | Python Fire interface for indexing, searching, answering, and evaluation. |
| **API** | `src/api/` | FastAPI service exposing `/health`, `/search`, and `/answer`. |

---

## 2. Chunking Strategy

- **Format-Aware Splitting:**
  - **Python (`.py`):** Uses `PythonCodeTextSplitter` to respect class, method, and function boundaries.
  - **Markdown & Text (`.md`, `.txt`):** Uses `RecursiveCharacterTextSplitter` splitting on double newlines and paragraphs.
- **Configuration & Boundaries:**
  - Maximum chunk size: **2,000 characters** (configurable via `--max_chunk_size`).
  - Chunk overlap: **0 characters** (prevents duplicate text in prompts).
  - Character offsets: Every chunk records `first_character_index` and `last_character_index` to allow exact extraction from the raw source files.
- **Impact of Chunk Size:**
  - *Too small:* Loses surrounding syntactic context (e.g. separating a method from its class definition).
  - *Too large:* Dilutes keyword relevance scores and risks overflowing the LLM prompt context window.

---

## 3. Retrieval Method (BM25)

Retrieval uses the **BM25s** library with Lucene scoring parameters:
- **Term Frequency Saturation ($k_1 = 1.5$):** Prevents keyword stuffing; score contributions plateau as frequency rises.
- **Document Length Normalization ($b = 0.75$):** Penalizes overly long documents so concise, relevant chunks can compete fairly.

### Storage Layout (`data/processed/`)
- `bm25_index/`: Precomputed sparse matrix arrays (`.csc.index.npy`, `vocab.index.json`).
- `chunks_metadata.pkl`: Pickled list of `MinimalSource` objects aligned with matrix row indices.
- `file_manifest.json` & `chunks_by_file.pkl`: File content MD5 hashes and chunk cache used for incremental updates.

*Note:* File paths are prepended to the indexed chunk text (`f"{file_path} {chunk_text}"`), allowing queries to match against module names and directory paths (e.g. `openai/api_server.py`).

---

## 4. Performance Analysis

Evaluated against the private benchmark dataset (100 docs questions, 100 code questions) on CPU:

| Metric | Measured | Required Threshold | Result |
|---|---|---|---|
| **Full Indexing Time** | ~7 seconds | ≤ 300 seconds | **PASS** |
| **Retrieval Throughput (200 Qs)** | ~6 seconds | ≤ 90 seconds | **PASS** |
| **Docs Recall@5** | **87.0%** | ≥ 80.0% | **PASS** |
| **Code Recall@5** | **62.0%** | ≥ 50.0% | **PASS** |
| **Edge Cases (4 tests)** | 4/4 clean exits | No unhandled tracebacks | **PASS** |
| **Answer Quality (Qwen 0.6B)** | 2/3 satisfactory | ≥ 2/3 | **PASS** |

---

## 5. Design Decisions & Trade-Offs

1. **BM25s vs. TF-IDF / Embeddings:**
   - *Choice:* Selected BM25s for fast CPU sparse-matrix indexing and retrieval.
   - *Trade-off:* High speed (~7s indexing, ~6s retrieval), but purely lexical; it cannot match synonyms that do not appear verbatim.
2. **2,000 Character Chunk Size with 0 Overlap:**
   - *Choice:* Maximizes context per chunk within the evaluator's 2,000-character source boundary.
   - *Trade-off:* Zero overlap keeps the index lean and avoids duplicate results, but context crossing a boundary gets separated.
3. **Path Prepending in Indexed Text:**
   - *Choice:* Appending relative file paths to the indexed chunk body.
   - *Trade-off:* Improves recall for queries referencing specific files or features (e.g. `quantization`), at the cost of slightly larger vocabulary.
4. **Incremental Indexing Mechanism:**
   - *Choice:* Uses MD5 hashes to re-chunk only modified files, reusing unchanged chunks.
   - *Trade-off:* Re-chunking is instant (0.01s), but the BM25 matrix is refreshed (0.15s) because Inverse Document Frequency (IDF) relies on global corpus statistics.

---

## 6. Example Usage

### Setup & Quality Checks
```bash
make install       # uv sync
make lint          # flake8 + mypy checks
```

### Basic Commands
```bash
# 1. Build the index
uv run python -m src index --corpus_dir data/raw --max_chunk_size 2000

# 2. Search top-k sources
uv run python -m src search "How to configure OpenAI server?" --k 5

# 3. Generate an answer
uv run python -m src answer "How to configure OpenAI server?" --k 5
```

### Dataset Evaluation Workflow
```bash
# Search over a dataset
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results --k 5

# Generate answers for search results
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer

# Evaluate Recall@k against ground truth
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json --k 5
```

---

## Bonus Features

### 1. Caching (Index & Query Results)
- Reuses the serialized BM25 index on disk across CLI calls.
- Caches single-query `search` and `answer` results in `cache/query_cache.json`.
- Repeat queries print `[Cache HIT ⚡]`.
- Cache invalidates when the index is rebuilt, and can be cleared via:
  ```bash
  uv run python -m src clear_cache
  ```

### 2. Incremental Indexing
- Detects added, modified, and deleted files using MD5 hashes in `data/processed/file_manifest.json`.
- Re-chunks only changed files, reusing untouched chunks from `data/processed/chunks_by_file.pkl`.
- Run via:
  ```bash
  uv run python -m src index --incremental
  ```

### 3. Local HTTP API (FastAPI)
- REST API built with FastAPI and Pydantic validation.
- Run the server:
  ```bash
  uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
  ```
- Interactive Swagger docs at `http://127.0.0.1:8000/docs`.
- Endpoints:
  - `GET /health` → `{"status": "ok"}`
  - `POST /search` → `{"query": "...", "k": 5}`
  - `POST /answer` → `{"query": "...", "k": 5}`
