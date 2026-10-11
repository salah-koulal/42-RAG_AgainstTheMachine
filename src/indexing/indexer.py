"""Walks data/raw/, chunks every file, and persists the index to data/processed/"""
import pickle
from typing import List
from rank_bm25 import BM25Okapi

from src.io_utils import load_corpus, PROCESSED_DIR, RAW_DIR
from src.chunking.text_chunker import chunk_text
from src.chunking.python_chunker import chunk_python
from src.models import MinimalSource


def build_index(corpus_dir: str = str(RAW_DIR), chunk_size: int = 2000) -> None:
    files = load_corpus(corpus_dir)

    all_chunks: List[MinimalSource] = []
    corpus_texts: List[str] = []

    print(f"Loaded {len(files)} files. Chunking...")

    for file_path, content in files:
        if file_path.endswith(".py"):
            chunks = chunk_python(file_path, content, chunk_size)
        
        elif file_path.endswith((".md", ".txt")):
            chunks = chunk_text(file_path, content, chunk_size)
        
        else:
            continue

        # adding those chunks to our lists
        for c in chunks:
            all_chunks.append(c)
            chunk_text_content = content[c.first_character_index:c.last_character_index]
            corpus_texts.append(chunk_text_content)



    print(f"Generated {len(all_chunks)} chunks. Building BM25 index...")
    # tokenize the text bm25 expects a list of words

    tokenized_corpus = [text.lower().split() for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROCESSED_DIR / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)

    with open(PROCESSED_DIR / "chunks_metadata.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print("indexing complete")
    


