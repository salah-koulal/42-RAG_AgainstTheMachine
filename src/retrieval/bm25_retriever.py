"""BM25 implementation of the retriever interface."""

import pickle
from typing import List, cast
from src.models import MinimalSource
from src.io_utils import PROCESSED_DIR

class BM25Retriever:
    def __init__(self) -> None:
        """
        Load the BM25 index and chunk metadata from disk into memory.
        """
        with open(PROCESSED_DIR / "bm25_index.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        with open(PROCESSED_DIR / "chunks_metadata.pkl", "rb") as f:
            self.chunks_metadata = pickle.load(f)


    def retrieve(self, query: str, k: int = 5) -> List[MinimalSource]:
        """
        Takes a raw user query, searches the BM25 index, and returns the top-k chunks.
        """
        tokenized_query = query.lower().split()
        top_chunks = self.bm25.get_top_n(tokenized_query, self.chunks_metadata, n=k)
        return cast(List[MinimalSource], top_chunks)
