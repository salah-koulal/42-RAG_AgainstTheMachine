"""Behavior checks for disk caching and incremental chunk reuse."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.caching import query_cache
from src.indexing import indexer, manifest
from src.retrieval.bm25_retriever import BM25Retriever


def test_cache_tracks_index_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(query_cache, "CACHE_FILE", tmp_path / "cache.json")
    monkeypatch.setattr(query_cache, "PROCESSED_DIR", tmp_path)
    query_cache.set_cached("search", " Hello ", 3, [])
    assert query_cache.get_cached("search", "hello", 3) == []
    assert query_cache.get_cached("search", "hello", 4) is None
    assert query_cache.get_cached("answer", "hello", 3) is None
    (tmp_path / "chunks_metadata.pkl").write_bytes(b"new index revision")
    assert query_cache.get_cached("search", "hello", 3) is None
    query_cache.CACHE_FILE.write_text("invalid json")
    assert query_cache.get_cached("search", "hello", 3) is None


def test_incremental_changes_reuse_unchanged_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    processed = tmp_path / "processed"
    monkeypatch.setattr(indexer, "PROCESSED_DIR", processed)
    monkeypatch.setattr(indexer, "CHUNKS_BY_FILE", processed / "chunks_by_file.pkl")
    monkeypatch.setattr(manifest, "MANIFEST_FILE", processed / "file_manifest.json")
    first = corpus / "first.md"
    second = corpus / "second.py"
    first.write_text("# Apples\nApples are red.\n")
    second.write_text('def banana():\n    return "yellow"\n')
    indexer.build_index(str(corpus), max_chunk_size=100)
    with patch.object(indexer, "chunk_file", wraps=indexer.chunk_file) as chunk:
        indexer.build_index_incremental(str(corpus), max_chunk_size=100)
        chunk.assert_not_called()
        first.write_text("# Oranges\nOranges contain citrus.\n")
        indexer.build_index_incremental(str(corpus), max_chunk_size=100)
        assert chunk.call_count == 1
        assert chunk.call_args.args[0] == str(first)
    retriever = BM25Retriever(processed)
    assert retriever.retrieve("citrus", k=1)[0].file_path == str(first)
    first.unlink()
    indexer.build_index_incremental(str(corpus), max_chunk_size=100)
    retriever = BM25Retriever(processed)
    assert all(source.file_path == str(second) for source in retriever.chunks_metadata)
    assert all(
        0 < source.last_character_index - source.first_character_index <= 100
        for source in retriever.chunks_metadata
    )
