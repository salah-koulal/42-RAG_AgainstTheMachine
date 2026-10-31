"""Unit test for incremental indexing."""

from src.indexing.indexer import build_index_incremental
from src.retrieval.bm25_retriever import BM25Retriever


def test_incremental_no_changes() -> None:
    """Verify that running incremental indexing with no changes completes cleanly."""
    build_index_incremental()
    retriever = BM25Retriever()
    results = retriever.retrieve("OpenAI server", k=2)
    assert len(results) > 0
