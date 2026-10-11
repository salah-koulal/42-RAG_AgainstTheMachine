"""Walks data/raw/, chunks every file, and persists the index to data/processed/."""

import pickle
from pathlib import Path
from typing import List, Optional

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.chunking.python_chunker import chunk_python
from src.chunking.text_chunker import chunk_text
from src.io_utils import PROCESSED_DIR, RAW_DIR, load_corpus
from src.models import MinimalSource
from src.retrieval.bm25_retriever import tokenize


def build_index(
    corpus_dir: str = str(RAW_DIR),
    max_chunk_size: int = 2000,
    chunk_size: Optional[int] = None,
) -> None:
    """Ingest corpus directory, chunk all files, and build the BM25 index."""
    effective_chunk_size = chunk_size if chunk_size is not None else max_chunk_size
    files = load_corpus(corpus_dir)

    all_chunks: List[MinimalSource] = []
    corpus_texts: List[str] = []

    print(f"Loaded {len(files)} files. Chunking...")

    for file_path, content in tqdm(files, desc="Chunking files", unit="file"):
        full_path = str(Path(corpus_dir) / file_path)
        if file_path.endswith(".py"):
            chunks = chunk_python(full_path, content, effective_chunk_size)
        elif file_path.endswith((".md", ".txt")):
            chunks = chunk_text(full_path, content, effective_chunk_size)
        else:
            continue

        for c in chunks:
            all_chunks.append(c)
            raw_chunk = content[c.first_character_index:c.last_character_index]
            # Prepend file path to chunk text for enhanced lexical recall
            indexed_text = f"{c.file_path} {raw_chunk}"
            corpus_texts.append(indexed_text)

    print(f"Generated {len(all_chunks)} chunks. Building BM25 index...")
    tokenized_corpus = [tokenize(text) for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROCESSED_DIR / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)

    with open(PROCESSED_DIR / "chunks_metadata.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print("Ingestion complete! Indices saved under data/processed/")
