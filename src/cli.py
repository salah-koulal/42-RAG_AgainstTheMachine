"""Fire-facing CLI: index, search, search_dataset, answer, answer_dataset, evaluate (Task 4/7)."""
from src.io_utils import load_corpus


class RagCLI:
    def index(self, corpus_dir: str, chunk_size: int = 2000) -> None:
        """"""
        print(corpus_dir, chunk_size)
        files = load_corpus(corpus_dir)
        print(f"Found {len(files)} files")
        for path, content in files[:3]:
            print(f"  {path} ({len(content)} chars)")

    
    def search(self, query: str, k: int = 5) -> None:
        print(query, k)

    def search_dataset(self, dataset_path: str, k: int = 3) -> None:
        print(dataset_path, k)

    def answer(self, query: str, k: int = 5) -> None:
        print(query, k)

    def answer_dataset(self, dataset_path: str, k: int = 5) -> None:
        print(dataset_path, k)

    def evaluate(self, dataset_path: str, search_results_path: str) -> None:
        print(dataset_path, search_results_path)
