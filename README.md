*This project has been created as part of the 42 curriculum by skoulal.*

# RAG against the machine

## Description

A local retrieval-augmented generation (RAG) system for the supplied vLLM 0.10.1
codebase and documentation. It finds relevant source locations with BM25 and feeds
their text to Qwen/Qwen3-0.6B to answer questions. Indexing prepares searchable data;
it does not train the language model. Both a Python Fire CLI and a local FastAPI
service expose the pipeline.

The project implements lexical retrieval. Semantic embeddings and hybrid retrieval
are outside the selected bonus scope. The three selected bonuses are disk caching,
incremental indexing, and a local HTTP API; the incremental implementation's
limitations are described below.

## Instructions

Run commands from the repository root so relative source paths remain valid.
Requirements: Python 3.10 or later, `uv`, and enough disk space for the Python
environment, corpus and model weights. The lockfile selects the concrete dependency
versions. CPU operation is supported; the generator selects CUDA if available.

```bash
make install
```

Place the supplied vLLM corpus at `data/raw/vllm-0.10.1/`. Place the supplied public
JSON datasets under `data/datasets/UnansweredQuestions/` and
`data/datasets/AnsweredQuestions/`. Corpus files, model weights and generated indexes
are not source code and should not be committed.

```bash
# Build an index before searching or starting the API.
uv run python -m src index --corpus_dir data/raw --max_chunk_size 2000
uv run python -m src search "What hardware platforms does vLLM support?" --k 3
uv run python -m src answer "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" --k 3
```

The first answer/API startup can download Qwen weights from Hugging Face. Later runs
reuse the local model cache. The API loads the model and index at startup; restart
it after rebuilding the index. Do not modify the corpus while serving requests.

### Dataset workflow

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results --k 10

uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer

uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json --k 5
```

Use the corresponding code dataset filename to evaluate code retrieval. The local
`evaluate` command is a development aid. Official results come from the supplied
moulinette, invoked externally by the exam scripts; application code does not import
or invoke it. Dataset output files preserve the input basename.

### Development commands

```bash
make run ARGS='search "What is vLLM?" --k 3'
make debug ARGS='search "What is vLLM?" --k 3'
make lint
uv run pytest -q
make clean
```

`make clean` removes Python/tool caches, not corpus files or model weights.

## System architecture

```text
.py / .md / .txt files
    -> format-specific chunking
    -> tokenize file path + chunk text
    -> BM25 index + chunk metadata on disk

question -> tokenization -> BM25 ranking -> top-k source locations
                                              |
                         read original character ranges
                                              |
                               context + question -> Qwen -> answer
```

| Module | Responsibility |
| --- | --- |
| `src/chunking/` | Split files and preserve character offsets |
| `src/indexing/` | Build BM25, save metadata, detect changed files |
| `src/retrieval/` | Load the saved index and retrieve source locations |
| `src/generation/` | Read snippets, build context, generate an answer |
| `src/caching/` | Persist single-query CLI results |
| `src/evaluation/` | Compute source-overlap recall |
| `src/cli.py` | Python Fire commands and dataset workflows |
| `src/api/main.py` | HTTP health, search and answer endpoints |

Pydantic models in `src/models.py` define the dataset and result contracts. A
`MinimalSource` contains `file_path`, `first_character_index` and
`last_character_index`. The character range is start-inclusive and end-exclusive.
Paths must match the evaluation corpus paths exactly.

## Chunking strategy

Python uses LangChain's Python-specific recursive separators. Markdown and plain
text use recursive character splitting. Both preserve source offsets and use zero
chunk overlap. The default maximum is 2,000 characters, matching the evaluator's
source-length limit. Use a positive size no larger than 2,000 for evaluated runs;
the CLI currently checks positivity but does not enforce the upper bound.

Smaller chunks reduce unrelated context but can separate a question's evidence.
Larger chunks retain more context but can dilute lexical scores. No chunk-size
ablation is claimed; the reported results use 2,000 characters.

## Retrieval method

BM25S stores a vocabulary and a sparse matrix of precomputed term–chunk score
contributions. Scores account for term frequency, term rarity and chunk length.
The implementation uses BM25S's Lucene method with `k1=1.5` and `b=0.75`.

`data/processed/bm25_index/` contains:

- `vocab.index.json`: tokens mapped to index IDs.
- `data.csc.index.npy`: score contributions.
- `indices.csc.index.npy`: corresponding chunk IDs.
- `indptr.csc.index.npy`: boundaries of each token's entries in the arrays.
- `params.index.json`: scoring settings, number of chunks and storage information.

`chunks_metadata.pkl` separately maps chunk IDs to source locations. It must stay
aligned with the index. Indexing also saves `chunks_by_file.pkl` and
`file_manifest.json` for incremental chunk reuse. Only load trusted local pickle
artifacts produced by this project.

The indexed text includes each chunk's path, making filenames and directory names
searchable. Query tokenization retains stopwords, while indexing uses BM25S's default
stopword handling. Retrieval maps query vocabulary to index vocabulary and sums
contributions before selecting top-k chunks. Scores are ranking values, not answer
confidence. Lexical matches can miss paraphrases or retrieve a related but narrower
topic. `k` is capped to the number of indexed chunks by the retriever.

## Answer generation

The generator reads retrieved character spans from the corpus and labels the
snippets with their paths. A system prompt instructs Qwen to answer from that
context and acknowledge insufficient evidence. Generation disables sampling and
thinking mode, and allows at most 256 new tokens. These settings improve
repeatability but do not guarantee factual correctness or complete answers.

## Selected bonuses

### Caching

The serialized BM25 index is reused between CLI runs. Single-query CLI `search`
and `answer` results are stored in `cache/query_cache.json`; repeat requests print
`[Cache HIT]` (with a lightning icon). Keys include mode, normalized query, k, and
an index revision derived from index/metadata file timestamps and sizes. Rebuilding
the index makes old entries ineligible. Empty result lists are valid cache hits.
Old revision entries remain on disk until cleared:

```bash
uv run python -m src clear_cache
```

This is a simple local JSON cache, not a concurrent multi-process cache. Changes to
model settings or source files without reindexing require clearing it manually.
Dataset commands bypass this query cache. The HTTP API deliberately does not use it.

### Incremental indexing: current limitation

```bash
uv run python -m src index --incremental True --max_chunk_size 2000
```

Content hashes detect new, changed and deleted files. Unchanged chunks are reused;
an unchanged corpus skips rebuilding. Modified files alone are re-chunked, but the
current implementation then rebuilds the complete BM25 matrix to refresh
corpus-wide statistics. Thus it demonstrates incremental ingestion/chunking, not
strict file-only index updates. Do not claim this bonus fully satisfies the subject's
literal requirement until that limitation is resolved and reviewed.

Use a full build when changing chunk size, corpus root or tokenization settings:
these configuration changes are not currently part of the manifest. Removing every
file is also an unverified edge case for incremental rebuilding.

### Local HTTP API

```bash
uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Interactive documentation: <http://127.0.0.1:8000/docs>.

| Method | Path | Response |
| --- | --- | --- |
| GET | `/health` | `{"status":"ok"}` |
| POST | `/search` | `query` and `results` (source locations) |
| POST | `/answer` | `query`, generated `answer` and `sources` |

Both POST routes accept `query` and optional `k` (default 5, range 1–50).
Queries are trimmed and must contain at least one non-whitespace character.
Malformed JSON, missing queries, blank queries and out-of-range k produce HTTP 422.

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"What hardware platforms does vLLM support?","k":3}'
curl -s http://127.0.0.1:8000/answer \
  -H 'Content-Type: application/json' \
  -d '{"query":"What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?","k":3}'
```

## Performance analysis and verification

Measured on the current local environment on 2026-10-31, using the supplied private
exam datasets (100 docs questions and 100 code questions). Timings are specific to
this machine and its existing dependency/model caches.

| Check | Result | Required |
| --- | --- | --- |
| Full indexing | 7 seconds | <= 300 seconds |
| Retrieval of 200 questions | 6 seconds | <= 90 seconds |
| Docs recall@5 | 87% | >= 80% |
| Code recall@5 | 62% | >= 50% |
| Retrieval exam | 4/4 passed | 4/4 |
| Edge-case exam | 4/4 passed | 4/4 |
| Answer exam (source-based review) | 2/3 satisfactory | >= 2/3 |
| API validation | 12 invalid requests rejected; search and answer succeeded | Local search and answering |
| Style and types | flake8 and mypy passed | No errors |
| Bonus regression tests | 2 passed | Cache correctness and chunk reuse |

The retrieval exam also reported valid output schemas and source lengths. Recall
measures the fraction of reference source spans retrieved, averaged over questions;
a match requires the same file and sufficient character-range IoU (0.05).

Retrieval evidence: `evaluations/retrieval/2026-10-31_12-25-52/`.
Edge-case evidence: `evaluations/edge_cases/2026-10-31_12-42-18/`.
Answer evidence: `evaluations/answer/2026-10-31_12-46-10/`.
The metric-name and latency-command answers matched source evidence. The GGUF answer
omitted the required single-file restriction and was judged incomplete. This is a
local review, not an official corrector's decision. API checks exercised real BM25
retrieval and Qwen generation through the ASGI application. All 14,313 indexed source
spans across 1,845 files were checked for valid bounds and the 2,000-character limit.
A one-question `answer_dataset` smoke test also passed: the saved result validated
against `StudentSearchResultsAndAnswer`, retained the question ID and input basename,
and contained an answer. This is not a full answer-dataset quality assessment.
The local recall implementation matched the moulinette's 87% and 62% results.
The edge-case script's missing-path command omits `save_directory`, so that particular
check only exercises CLI argument validation. A separate complete command is needed
to exercise the missing-file branch.

Reproduce the supplied exams (private datasets and moulinette are provided separately):

```bash
bash exams/scripts/exam_retrieval.sh --student-path . --moulinette-path ./moulinette
bash exams/scripts/exam_edge_cases.sh --student-path .
bash exams/scripts/exam_answer.sh --student-path . --moulinette-path ./moulinette
```

The answer exam asks the reviewer to select three valid questions and judge whether
at least two answers are satisfactory. A successful script exit alone does not
establish answer quality. The exam scripts do not grade the three bonuses.

Before submission, review tracked generated artifacts: the current repository
already tracks query-cache data, historical evaluation output and private datasets.
Adding ignore rules alone does not untrack them. Keep the required source and
documentation, and follow the subject's rules for supplied/generated data.

## Design decisions and challenges

- Preserve exact paths and character ranges so retrieved sources can be evaluated
  and their original text can be recovered without duplicating it in responses.
- Use BM25 to keep retrieval small and CPU-friendly; lexical matching remains weaker
  for paraphrases than semantic retrieval.
- Keep index construction outside HTTP request handling. One retriever and generator
  are loaded per API process, so startup requires an existing index and model access.
- Separate retrieval from generation: source quality can be measured independently
  of the model's ability to follow context.
- Treat cache validity as tied to an index revision. Reusing old source ranges after
  rebuilding an index can silently produce incorrect context.
- Distinguish changed-file chunk reuse from a genuinely incremental BM25 update:
  global term statistics make the latter more involved.

## Example usage

For `search "What hardware platforms does vLLM support?" --k 3`, the inspected
index returned:

```text
#1 - data/raw/vllm-0.10.1/docs/features/quantization/supported_hardware.md [2059:2697]
#2 - data/raw/vllm-0.10.1/docs/features/quantization/supported_hardware.md [0:219]
#3 - data/raw/vllm-0.10.1/tests/quantization/test_fp8.py [5248:6672]
```

These are source locations, not a generated answer. They illustrate a retrieval
limitation: the question is broad, while the top results focus on quantization.
For the LoRA endpoint question, the API demonstration returned `/v1/load_lora_adapter`
with source locations from the corpus.

## Resources

- BM25S implementation and documentation: <https://github.com/xhluca/bm25s>
- Qwen3-0.6B model card: <https://huggingface.co/Qwen/Qwen3-0.6B>
- LangChain text splitters: <https://docs.langchain.com/oss/python/integrations/splitters>
- Pydantic documentation: <https://docs.pydantic.dev/latest/>
- FastAPI documentation: <https://fastapi.tiangolo.com/>
- Python Fire: <https://github.com/google/python-fire>
- uv documentation: <https://docs.astral.sh/uv/>
- The supplied project subject, vLLM corpus and external exam scripts.

AI assistance was used for guided FastAPI exercises, code review, explanations of
BM25 indexing and retrieval, debugging request validation, running verification,
cache correctness fixes, and drafting this README. The author remains responsible
for understanding and explaining the implementation and checking generated answers
against source evidence.
