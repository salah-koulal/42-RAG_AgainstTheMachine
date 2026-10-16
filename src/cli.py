"""Command-Line Interface built with Python Fire (Task 4/7)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.generation.generator import QwenGenerator
from src.indexing.indexer import build_index
from src.io_utils import RAW_DIR
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.retrieval.bm25_retriever import BM25Retriever


class RagCLI:
    """CLI commands exposing indexing, search, generation, and evaluation."""

    def index(
        self,
        corpus_dir: str = str(RAW_DIR),
        max_chunk_size: int = 2000,
        chunk_size: Optional[int] = None,
    ) -> None:
        """Ingest data/raw/ and build the index under data/processed/."""
        corpus_path = Path(corpus_dir)
        if not corpus_path.exists():
            print(f"Error: Corpus directory not found at {corpus_dir}")
            return
        effective_chunk_size = chunk_size if chunk_size is not None else max_chunk_size
        if effective_chunk_size <= 0:
            print("Error: max_chunk_size must be positive.")
            return

        build_index(
            corpus_dir=corpus_dir,
            max_chunk_size=effective_chunk_size,
        )

    def search(self, query: str = "", k: int = 5) -> None:
        """Return the top-k sources for a single query."""
        if not query or not query.strip():
            print("Empty query provided. Returning 0 results.")
            return
        if k <= 0:
            print(f"Top {k} results: None (k must be positive).")
            return

        try:
            retriever = BM25Retriever()
        except Exception as e:
            print(f"Error loading index: {e}. Please run index first.")
            return

        results = retriever.retrieve(query, k=k)
        print(f"Top {len(results)} results for: {query}:")
        for rank, res in enumerate(results, 1):
            print(
                f"#{rank} - {res.file_path} "
                f"[{res.first_character_index}:{res.last_character_index}]"
            )

    def search_dataset(
        self, dataset_path: str, save_directory: str, k: int = 5
    ) -> None:
        """Run search over a dataset and write a StudentSearchResults JSON file."""
        if k <= 0:
            print("Error: k must be a positive integer.")
            return

        p = Path(dataset_path)
        if not p.is_file():
            print(f"Error: Dataset file not found at {dataset_path}")
            return

        try:
            with open(p, "r", encoding="utf-8") as f:
                dataset: Dict[str, Any] = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Malformed JSON in {dataset_path}")
            return
        except Exception as e:
            print(f"Error reading {dataset_path}: {e}")
            return

        questions = dataset.get("rag_questions")
        if not isinstance(questions, list):
            print(f"Error: Missing or invalid rag_questions in {dataset_path}")
            return

        try:
            retriever = BM25Retriever()
        except Exception as e:
            print(f"Error loading index: {e}. Please run index first.")
            return

        all_results: List[MinimalSearchResults] = []
        for q in tqdm(questions, desc="Searching questions", unit="query"):
            q_text = q.get("question", "") if isinstance(q, dict) else ""
            q_id = q.get("question_id", "") if isinstance(q, dict) else ""
            top_chunks = retriever.retrieve(q_text, k=k)
            all_results.append(
                MinimalSearchResults(
                    question_id=q_id,
                    question=q_text,
                    retrieved_sources=top_chunks,
                )
            )

        final_output = StudentSearchResults(search_results=all_results, k=k)
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / p.name
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(final_output.model_dump_json(indent=2))

        print(f"Saved {len(all_results)} search results to {out_file}")

    def answer(self, query: str = "", k: int = 5) -> None:
        """Answer a single query using retrieved context."""
        if not query or not query.strip():
            print("Empty query provided.")
            return
        if k <= 0:
            print("Error: k must be a positive integer.")
            return

        try:
            retriever = BM25Retriever()
        except Exception as e:
            print(f"Error loading index: {e}. Please run index first.")
            return

        top_chunks = retriever.retrieve(query, k=k)
        try:
            generator = QwenGenerator()
            ans = generator.generate_answer(query, top_chunks)
        except Exception as e:
            print(f"Error generating answer: {e}")
            return

        print("\n================ ANSWER ================")
        print(ans)
        print("========================================\n")

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """Generate answers for a dataset, producing StudentSearchResultsAndAnswer."""
        p = Path(student_search_results_path)
        if not p.is_file():
            print(f"Error: File not found at {student_search_results_path}")
            return

        try:
            with open(p, "r", encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Malformed JSON in {student_search_results_path}")
            return
        except Exception as e:
            print(f"Error reading {student_search_results_path}: {e}")
            return

        search_results = data.get("search_results", [])
        if not isinstance(search_results, list):
            print("Error: search_results must be a list.")
            return

        k = int(data.get("k", 5))
        print(f"Loaded {len(search_results)} questions ...")

        try:
            generator = QwenGenerator()
        except Exception as e:
            print(f"Error initializing model generator: {e}")
            return

        answered: List[MinimalAnswer] = []
        for item in tqdm(search_results, desc="Generating answers", unit="question"):
            search_item = MinimalSearchResults(**item)
            ans_text = generator.generate_answer(
                search_item.question, search_item.retrieved_sources
            )
            answered.append(
                MinimalAnswer(
                    question_id=search_item.question_id,
                    question=search_item.question,
                    retrieved_sources=search_item.retrieved_sources,
                    answer=ans_text,
                )
            )

        output_data = StudentSearchResultsAndAnswer(search_results=answered, k=k)
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / p.name

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(output_data.model_dump_json(indent=2))

        print(f"Saved student_search_results_and_answer to {out_file}")

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str = "",
        ground_truth_path: str = "",
        k: Optional[int] = None,
    ) -> None:
        """Report recall@k against a ground-truth dataset."""
        target_gt = dataset_path if dataset_path else ground_truth_path
        if not target_gt:
            print("Error: Please provide --dataset_path <path>")
            return

        p_stud = Path(student_search_results_path)
        p_gt = Path(target_gt)

        if not p_stud.is_file():
            print(f"Error: Student search results file not found at {p_stud}")
            return
        if not p_gt.is_file():
            print(f"Error: Ground truth dataset file not found at {p_gt}")
            return

        try:
            with open(p_stud, "r", encoding="utf-8") as f:
                student_data: Dict[str, Any] = json.load(f)
            with open(p_gt, "r", encoding="utf-8") as f:
                gt_data: Dict[str, Any] = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Malformed JSON file: {e}")
            return
        except Exception as e:
            print(f"Error reading JSON files: {e}")
            return

        student_results = student_data.get("search_results", [])
        gt_questions = gt_data.get("rag_questions", [])

        from src.evaluation.metrics import compute_recall_at_k

        total_cnt = len(student_results)
        if total_cnt == 0:
            print("Warning: No search results found to evaluate.")
            return

        print("\n========================================")
        if k is not None:
            recall = compute_recall_at_k(student_results, gt_questions, k=k)
            print(f"  Recall@{k}: {recall:.2%}")
            print(f"  ({recall.hits}/{recall.total} questions hit)")
        else:
            file_k = int(student_data.get("k", 5))
            cutoffs = [c for c in [1, 3, 5, 10] if c <= file_k]
            if file_k not in cutoffs:
                cutoffs.append(file_k)
            cutoffs = sorted(set(cutoffs))

            print("  Evaluation Results:")
            for cutoff in cutoffs:
                rec = compute_recall_at_k(student_results, gt_questions, k=cutoff)
                tag = " (Subject Requirement)" if cutoff == 5 else ""
                print(
                    f"  Recall@{cutoff:02d}: {rec:6.2%} "
                    f"({rec.hits}/{rec.total} questions hit){tag}"
                )
        print("========================================\n")
