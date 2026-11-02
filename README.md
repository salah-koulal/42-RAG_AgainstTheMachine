*This project has been created as part of the 42 curriculum by skoulal.*

# RAG against the machine

## Description

A local Retrieval-Augmented Generation (RAG) system built to query and answer questions over the **vLLM (0.10.1)** codebase and documentation. 

Instead of relying on the base knowledge of a large language model or costly fine-tuning, this system:
1. Ingests and chunks source code (`.py`) and documentation (`.md`, `.txt`).
2. Indexes text into a sparse lexical matrix using **BM25**.
3. Retrieves relevant code snippets using exact file-character offsets.
4. Augments the prompt of **Qwen/Qwen3-0.6B** to produce accurate, hallucination-free answers grounded strictly in the codebase.

---

## Instructions & Setup

### Prerequisites
- **Python**: Version 3.10 or higher.
- **Package Manager**: [`uv`](https://docs.astral.sh/uv/) is used for all environment and dependency management.
- **Hardware**: Fully functional on standard campus CPU machines (CUDA is automatically used if available).

### Installation
Run the following from the repository root:
```bash
make install
```
*(This executes `uv sync` to install all locked dependencies from `uv.lock` into `.venv`)*.

### Data Placement
Before running the pipeline, ensure the following directories are populated:
- **Corpus:** Unzip `vllm-0.10.1.zip` into `data/raw/vllm-0.10.1/`.
- **Public Datasets:** Place the JSON datasets in `data/datasets/UnansweredQuestions/` and `data/datasets/AnsweredQuestions/`.

> **Note:** Raw corpus files, model weights, and generated outputs are excluded from git tracking via `.gitignore`.

---

## 1. System Architecture

### Pipeline Flow

```text
[Raw Documents: .py, .md, .txt]
              │
              ▼
  [Format-Aware Chunking] ──► (Track file_path + start/end character offsets)
              │
              ▼
   [BM25 Indexing (bm25s)] ───► Stored in data/processed/ (Index + Metadata)
              │
[User Query] ─┴─► [BM25 Scoring & Retrieval] ──► [Top-k MinimalSource Spans]
                                                            │
                                                            ▼
                                              [Read Text Slices from Disk]
                                                            │
                                                            ▼
[Qwen3-0.6B Prompt Augmented with Context] ────────► [Final Grounded Answer]
```

### Components and Interactions

| Component | Directory / File | Description & Interaction |
|---|---|---|
| **Chunking** | `src/chunking/` | Splits `.py` files by AST/code boundaries and `.md`/`.txt` by paragraphs while recording exact character offsets. |
| **Indexing** | `src/indexing/` | Builds the BM25 index, maps metadata, and handles incremental change detection via file hashing. |
| **Retrieval** | `src/retrieval/` | Tokenizes user queries, queries the BM25 matrix, and returns top-$k$ ranked source spans. |
| **Generation** | `src/generation/` | Slices text using offsets, builds an augmented prompt, and runs local inference with Qwen3-0.6B. |
| **Evaluation** | `src/evaluation/` | Computes source IoU (Intersection over Union) and Recall@$k$ against ground-truth datasets. |
| **Caching** | `src/caching/` | Provides disk-based caching for single-query CLI searches and answers. |
| **CLI** | `src/cli.py` | Exposes commands via Python Fire for all lifecycle actions. |
| **HTTP API** | `src/api/main.py` | Exposes REST endpoints (`/health`, `/search`, `/answer`) using FastAPI and Pydantic. |

---

## 2. Chunking Strategy

To handle both programming logic and human documentation effectively, two distinct chunking strategies are used:

1. **Python Files (`.py`):**
   - Implemented via `PythonCodeTextSplitter`.
   - Splits on language-specific boundaries (functions, classes, methods) to avoid breaking semantic code blocks.
2. **Markdown and Text Files (`.md`, `.txt`):**
   - Implemented via `RecursiveCharacterTextSplitter`.
   - Splits on double newlines, headers, and paragraphs before resorting to single line breaks or spaces.

### Boundaries and Offset Preservation
- **Max Chunk Size:** Defaults to **2,000 characters** (configurable via `--max_chunk_size`), aligning with the maximum source limit.
- **Overlap:** Set to **0 characters** to prevent duplicate text from bloating the LLM prompt.
- **Exact Character Ranges:** Every chunk records `first_character_index` and `last_character_index` relative to the raw source file. This allows retrieving and displaying exact locations without storing redundant file text inside the index.

### Impact of Chunk Size on Retrieval Performance
- **If chunks are too small:** A function or class can be split across multiple chunks, losing essential context (variable declarations, imports, or docstrings) and degrading answer quality.
- **If chunks are too large:** Irrelevant code surrounding the target snippet dilutes lexical keyword scores, reducing ranking precision and risking context overflow during generation.

---

## 3. Retrieval Method (BM25)

The retrieval engine is built on **BM25s**, an optimized Python implementation of the Lucene-style BM25 algorithm.

### Algorithm & Scoring Parameters
BM25 ranks documents based on term frequency (TF) and inverse document frequency (IDF) with two core parameters:
- **$k_1 = 1.5$ (Term Frequency Saturation):** Controls non-linear term frequency scaling. Unlike TF-IDF where score increases linearly, BM25 ensures that a word appearing 50 times does not dominate the score over a word appearing 5 times.
- **$b = 0.75$ (Document Length Normalization):** Penalizes documents that are longer than the average chunk length, ensuring concise, highly relevant code chunks are not outranked simply by document length.

### Storage Layout (`data/processed/`)
- `bm25_index/`: Contains vocabulary (`vocab.index.json`) and precomputed sparse matrix arrays (`.csc.index.npy`, `indptr.csc.index.npy`, `indices.csc.index.npy`).
- `chunks_metadata.pkl`: Pickled list of `MinimalSource` objects indexed in matching matrix row order.
- `file_manifest.json` & `chunks_by_file.pkl`: File content MD5 hashes and chunk cache used for incremental re-indexing.

### Path-Aware Search
During indexing, the relative file path is prepended to the chunk text (`f"{file_path} {chunk_text}"`). This ensures that queries specifying directory names, module names, or filenames (e.g. `quantization`, `api_server.py`) receive strong relevance boosts.

---

## 4. Performance Analysis

All benchmarks were measured on a campus CPU environment using the private evaluation datasets (100 documentation questions and 100 code questions):

| Metric / Test | Measured Result | Required Threshold | Status |
|---|---|---|---|
| **Full Indexing Time** | **~7 seconds** | ≤ 300 seconds (5 min) | **PASS** |
| **Retrieval Throughput (200 Qs)** | **~6 seconds** | ≤ 90 seconds | **PASS** |
| **Documentation Recall@5** | **87.0%** | ≥ 80.0% | **PASS** |
| **Code Recall@5** | **62.0%** | ≥ 50.0% | **PASS** |
| **System Reliability (Edge Cases)** | 4/4 passed | No unhandled tracebacks | **PASS** |
| **Answer Quality (Qwen 0.6B)** | 2/3 satisfactory | ≥ 2/3 | **PASS** |

### Evaluation Methodology
- **Recall Calculation:** A retrieved source counts as a match if it points to the correct file and has a 1D character Intersection over Union (IoU) $\ge 0.05$ with the ground-truth span.
- **Edge Cases Tested:** Handled empty query `""`, gibberish query `"asdfghjkl"`, $k=0$, and nonexistent dataset paths without any crashing tracebacks.

---

## 5. Design Decisions & Trade-Offs

1. **BM25s Sparse Index vs. Basic TF-IDF / Neural Embeddings:**
   - *Decision:* Used `bm25s` with precomputed `.csc` arrays.
   - *Trade-off:* Delivers fast indexing (7s) and search (6s) on CPU, but remains lexical; queries with paraphrasing or synonyms that do not exist in the code require exact token overlap.
2. **2,000 Character Chunk Size with 0 Overlap:**
   - *Decision:* Configured max chunk size to 2,000 chars without overlap.
   - *Trade-off:* Avoids redundant duplicated context in the LLM context window, but definitions spanning boundary splits can occasionally be divided.
3. **Path Prepending in Indexed Text:**
   - *Decision:* Appending relative file paths to the chunk text.
   - *Trade-off:* Drastically improves search precision for queries naming files or components, with a negligible increase in vocabulary size.
4. **Incremental Indexing via Content Hashing:**
   - *Decision:* Detect modified/new files using MD5 hashes to re-chunk only what changed.
   - *Trade-off:* Re-chunking runs in milliseconds (0.01s), but the BM25 matrix is recalculated in 0.15s to update global IDF statistics across the corpus.

---

## 6. Example Usage & Commands

### Development & Verification
```bash
make install     # Install locked dependencies (uv sync)
make lint        # Run flake8 and mypy type checks
make clean       # Remove caches and temporary files
```

### Core CLI Commands
```bash
# 1. Index the corpus
uv run python -m src index --corpus_dir data/raw --max_chunk_size 2000

# 2. Search top-k sources for a query
uv run python -m src search "How to configure OpenAI server?" --k 5

# 3. Generate a grounded answer
uv run python -m src answer "How to configure OpenAI server?" --k 5
```

### Dataset Batch Processing & Evaluation
```bash
# Batch search over unanswered questions
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results --k 5

# Batch answer generation from search results
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

### 1. Caching of Index & Query Results
- Reuses the loaded BM25 sparse index across invocations.
- Single-query CLI `search` and `answer` results are persisted in `cache/query_cache.json`.
- Cache keys track query text, $k$, mode, and index modification timestamps (auto-invalidating on index rebuilds). Repeat requests display `[Cache HIT ⚡]`.
- Clear cache via CLI:
  ```bash
  uv run python -m src clear_cache
  ```

### 2. Incremental Indexing
- Checks files against `data/processed/file_manifest.json` using MD5 hashes.
- Only re-chunks modified or new files, reusing existing chunks from `data/processed/chunks_by_file.pkl`.
- Run incremental update:
  ```bash
  uv run python -m src index --incremental
  ```

### 3. Local HTTP API (FastAPI)
- Exposes retrieval and generation over HTTP with full Pydantic request/response validation.
- Launch the API server:
  ```bash
  uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
  ```
- Interactive Swagger documentation: `http://127.0.0.1:8000/docs`
- Endpoints:
  - `GET  /health` → `{"status": "ok"}`
  - `POST /search` → Input `{"query": "...", "k": 5}`, returns source spans.
  - `POST /answer` → Input `{"query": "...", "k": 5}`, returns answer and source spans.

---

## Resources & References

- **BM25s Library**: [https://github.com/xhluca/bm25s](https://github.com/xhluca/bm25s)
- **Qwen 2.5 / 3 LLM Series**: [https://huggingface.co/Qwen](https://huggingface.co/Qwen)
- **LangChain Text Splitters**: [https://docs.langchain.com/oss/python/integrations/splitters](https://docs.langchain.com/oss/python/integrations/splitters)
- **FastAPI Documentation**: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
- **Pydantic Validation**: [https://docs.pydantic.dev/latest/](https://docs.pydantic.dev/latest/)
- **Astral uv Package Manager**: [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)

### AI Usage Disclosure
Generative AI was utilized during project development for understanding BM25 mathematical formulations, guiding FastAPI integration, refining edge case handling, and structuring documentation. All code was reviewed, validated, and tested for compliance with the 42 curriculum requirements.
