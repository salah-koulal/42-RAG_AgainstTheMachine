"""Walks data/raw/, chunks files, and persists the BM25 index to data/processed/."""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import bm25s
from tqdm import tqdm

from src.chunking.python_chunker import chunk_python
from src.chunking.text_chunker import chunk_text
from src.indexing.manifest import compute_file_hash, load_manifest, save_manifest
from src.io_utils import PROCESSED_DIR, RAW_DIR, load_corpus
from src.models import MinimalSource

CHUNKS_BY_FILE = PROCESSED_DIR / "chunks_by_file.pkl"


def chunk_file(
    full_path: str,
    content: str,
    max_chunk_size: int,
) -> List[Tuple[MinimalSource, str]]:
    """Chunk a single file and return pairs of (MinimalSource, indexed_text)."""
    if full_path.endswith(".py"):
        chunks = chunk_python(full_path, content, max_chunk_size)
    elif full_path.endswith((".md", ".txt")):
        chunks = chunk_text(full_path, content, max_chunk_size)
    else:
        return []

    items: List[Tuple[MinimalSource, str]] = []
    for c in chunks:
        raw_chunk = content[c.first_character_index:c.last_character_index]
        indexed_text = f"{c.file_path} {raw_chunk}"
        items.append((c, indexed_text))
    return items


def build_index_incremental(
    corpus_dir: str = str(RAW_DIR),
    max_chunk_size: int = 2000,
) -> None:
    """Incrementally update the index by re-chunking only modified or new files."""
    manifest = load_manifest()

    if not manifest or not CHUNKS_BY_FILE.is_file():
        print("No existing manifest/cache found. Performing initial full build...")
        build_index(corpus_dir=corpus_dir, max_chunk_size=max_chunk_size, incremental=False)
        return

    try:
        with open(CHUNKS_BY_FILE, "rb") as f:
            chunks_by_file: Dict[str, List[Tuple[MinimalSource, str]]] = pickle.load(f)
    except Exception:
        print("Cache corrupted. Performing full build...")
        build_index(corpus_dir=corpus_dir, max_chunk_size=max_chunk_size, incremental=False)
        return

    files = load_corpus(corpus_dir)
    current_map = {rel_path: content for rel_path, content in files}

    changed_files: List[Tuple[str, str]] = []
    for rel_path, content in files:
        cur_hash = compute_file_hash(content)
        if rel_path not in manifest or manifest[rel_path] != cur_hash:
            changed_files.append((rel_path, content))

    deleted_paths: Set[str] = set(manifest.keys()) - set(current_map.keys())

    if not changed_files and not deleted_paths:
        print(
            f"⚡ No files changed. Index is already up to date! "
            f"({len(files)} files unchanged)"
        )
        return

    print(
        f"Changes detected: {len(changed_files)} modified/new file(s), "
        f"{len(deleted_paths)} deleted file(s)."
    )

    for del_path in deleted_paths:
        manifest.pop(del_path, None)
        chunks_by_file.pop(del_path, None)

    for rel_path, content in changed_files:
        full_path = str(Path(corpus_dir) / rel_path)
        items = chunk_file(full_path, content, max_chunk_size)
        chunks_by_file[rel_path] = items
        manifest[rel_path] = compute_file_hash(content)

    all_chunks: List[MinimalSource] = []
    corpus_texts: List[str] = []
    for items in chunks_by_file.values():
        for c, text in items:
            all_chunks.append(c)
            corpus_texts.append(text)

    print(f"Rebuilding BM25 index with {len(all_chunks)} total chunks...")
    tokenized_corpus = bm25s.tokenize(corpus_texts, show_progress=False)
    bm25 = bm25s.BM25()
    bm25.index(tokenized_corpus)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    bm25.save(str(PROCESSED_DIR / "bm25_index"))

    with open(PROCESSED_DIR / "chunks_metadata.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    with open(CHUNKS_BY_FILE, "wb") as f:
        pickle.dump(chunks_by_file, f)

    save_manifest(manifest)
    print("Incremental indexing complete! Indices updated under data/processed/")


def build_index(
    corpus_dir: str = str(RAW_DIR),
    max_chunk_size: int = 2000,
    chunk_size: Optional[int] = None,
    incremental: bool = False,
) -> None:
    """Ingest corpus directory, chunk all files, and build the BM25 index."""
    effective_chunk_size = chunk_size if chunk_size is not None else max_chunk_size

    if incremental:
        build_index_incremental(corpus_dir=corpus_dir, max_chunk_size=effective_chunk_size)
        return

    files = load_corpus(corpus_dir)

    all_chunks: List[MinimalSource] = []
    corpus_texts: List[str] = []
    chunks_by_file: Dict[str, List[Tuple[MinimalSource, str]]] = {}
    manifest: Dict[str, str] = {}

    print(f"Loaded {len(files)} files. Chunking...")
    if not files:
        print(
            f"Error: No supported files (.py, .md, .txt) found in '{corpus_dir}'. "
            "Indexing aborted."
        )
        return

    for file_path, content in tqdm(files, desc="Chunking files", unit="file"):
        full_path = str(Path(corpus_dir) / file_path)
        items = chunk_file(full_path, content, effective_chunk_size)
        chunks_by_file[file_path] = items
        manifest[file_path] = compute_file_hash(content)

        for c, indexed_text in items:
            all_chunks.append(c)
            corpus_texts.append(indexed_text)

    print(f"Generated {len(all_chunks)} chunks. Building BM25 index...")
    if not all_chunks:
        print(
            f"Error: No chunks generated from files in '{corpus_dir}'. "
            "Indexing aborted."
        )
        return

    tokenized_corpus = bm25s.tokenize(corpus_texts, show_progress=False)
    bm25 = bm25s.BM25()
    bm25.index(tokenized_corpus)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    bm25.save(str(PROCESSED_DIR / "bm25_index"))

    with open(PROCESSED_DIR / "chunks_metadata.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    with open(CHUNKS_BY_FILE, "wb") as f:
        pickle.dump(chunks_by_file, f)

    save_manifest(manifest)

    print("Ingestion complete! Indices saved under data/processed/")
