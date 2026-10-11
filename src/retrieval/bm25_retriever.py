"""BM25 implementation of the retriever interface."""

import pickle
import re
from typing import List, cast

from src.io_utils import PROCESSED_DIR
from src.models import MinimalSource


def tokenize(text: str) -> List[str]:
    """Tokenize text by splitting camelCase and snake_case into individual terms."""
    s = re.sub(r"([a-z])([A-Z])", r"\\1 \\2", text)
    tokens = re.findall(r"[a-zA-Z0-9]+", s.lower())
    extra: List[str] = []
    for t in tokens:
        parts = t.split("_")
        if len(parts) > 1:
            extra.extend([p for p in parts if p])
    return tokens + extra


class BM25Retriever:
    """Retriever using BM25Okapi over code and documentation chunks."""

    def __init__(self) -> None:
        """Load the BM25 index and chunk metadata from disk into memory."""
        with open(PROCESSED_DIR / "bm25_index.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        with open(PROCESSED_DIR / "chunks_metadata.pkl", "rb") as f:
            self.chunks_metadata = pickle.load(f)

    def retrieve(self, query: str, k: int = 5) -> List[MinimalSource]:
        """Take a user query, search BM25 index, and return top-k chunks."""
        if not query or not query.strip() or k <= 0:
            return []
        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []
        top_chunks = self.bm25.get_top_n(tokenized_query, self.chunks_metadata, n=k)
        return cast(List[MinimalSource], top_chunks)
