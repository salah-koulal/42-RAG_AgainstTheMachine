"""BM25 implementation of the retriever interface using bm25s."""

import pickle
from pathlib import Path
from typing import List, Union

import bm25s

from src.io_utils import PROCESSED_DIR
from src.models import MinimalSource


class BM25Retriever:
    """Retriever using bm25s over code and documentation chunks."""

    def __init__(self, processed_dir: Union[str, Path] = str(PROCESSED_DIR)) -> None:
        """Load the BM25 index and chunk metadata from disk into memory."""
        p = Path(processed_dir)
        self.bm25 = bm25s.BM25.load(str(p / "bm25_index"), load_corpus=False)
        with open(p / "chunks_metadata.pkl", "rb") as f:
            self.chunks_metadata: List[MinimalSource] = pickle.load(f)

    def retrieve(self, query: str, k: int = 5) -> List[MinimalSource]:
        """Take a user query, search BM25 index, and return top-k chunks."""
        if not query or not query.strip() or k <= 0:
            return []
        query_tokens = bm25s.tokenize(query, show_progress=False, stopwords=[])
        actual_k = min(k, len(self.chunks_metadata))
        if actual_k == 0:
            return []
        results, _ = self.bm25.retrieve(query_tokens, k=actual_k, show_progress=False)
        indices = results[0].tolist()
        return [self.chunks_metadata[i] for i in indices]
