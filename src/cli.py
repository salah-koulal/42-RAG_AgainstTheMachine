"""Fire-facing CLI: index, search, search_dataset, answer, answer_dataset, evaluate (Task 4/7)."""
from src.io_utils import load_corpus
from tqdm import tqdm
import json
from pathlib import Path
from src.retrieval.bm25_retriever import BM25Retriever
from src.generation.generator import QwenGenerator
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)

class RagCLI:
    def index(self, corpus_dir: str, chunk_size: int = 2000) -> None:
        
        from src.indexing.indexer import build_index 
        build_index(corpus_dir,chunk_size)

    
    def search(self, query: str, k: int = 5) -> None:
        from src.retrieval.bm25_retriever import BM25Retriever
        
        print(f"Loading index and searching for: '{query}'...")
        retriever = BM25Retriever()
        results = retriever.retrieve(query, k)
        print(f"\nTop {k} results:")
        for rank, res in enumerate(results, 1):
            print(f"#{rank} - {res.file_path} (Chars: {res.first_character_index} to {res.last_character_index})")

    def search_dataset(self, dataset_path: str, save_directory: str, k: int = 5) -> None:
        import json
        from pathlib import Path
        from src.retrieval.bm25_retriever import BM25Retriever
        from src.models import StudentSearchResults, MinimalSearchResults

        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        retriever = BM25Retriever()
        all_results = []

        for q in tqdm(dataset["rag_questions"], desc="Searching questions", unit=" query"):
            top_chunks = retriever.retrieve(q["question"], k)
            
            single_result = MinimalSearchResults(
                question_id=q["question_id"],
                question=q["question"],
                retrieved_sources=top_chunks
            )

            all_results.append(single_result)

        final_output = StudentSearchResults(
            search_results=all_results,
            k=k
        )

        Path(save_directory).mkdir(parents=True, exist_ok=True)

        input_filename = Path(dataset_path).name
        output_path = Path(save_directory) / input_filename

        with open(output_path, "w") as f:
            f.write(final_output.model_dump_json(indent=4))
        
        print(f"Saved {len(all_results)} search results to {output_path}")
    def answer(self, query: str, k: int = 5) -> None:
        """Answer a single query using retrieved context."""
        retriever = BM25Retriever()
        top_chunks = retriever.retrieve(query, k=k)
        generator = QwenGenerator()
        ans = generator.generate_answer(query, top_chunks)

        print("\n================ ANSWER ================")
        print(ans)
        print("========================================\n")


    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """Generate answers for a dataset, producing a StudentSearchResultsAndAnswer JSON file."""
        input_path = Path(student_search_results_path)
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        search_results = data.get("search_results", [])
        k = data.get("k", 3)

        print(f"Loaded {len(search_results)} questions ...")

        generator = QwenGenerator()
        answered_results = []

        for item in tqdm(search_results, desc="Generating answers", unit="question"):
            # Reconstruct MinimalSearchResults object
            search_item = MinimalSearchResults(**item)
            
            # Generate the answer using the retrieved sources
            ans_text = generator.generate_answer(
                search_item.question, search_item.retrieved_sources
            )

            # Create MinimalAnswer
            answered_results.append(
                MinimalAnswer(
                    question_id=search_item.question_id,
                    question=search_item.question,
                    retrieved_sources=search_item.retrieved_sources,
                    answer=ans_text,
                )
            )

        output_data = StudentSearchResultsAndAnswer(
            search_results=answered_results, k=k
        )

        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / input_path.name

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(output_data.model_dump_json(indent=2))

        print(f"Saved student_search_results_and_answer to {out_file}")


    def evaluate(self, dataset_path: str, search_results_path: str) -> None:
        print(dataset_path, search_results_path)
