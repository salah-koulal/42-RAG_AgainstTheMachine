"""Shared helpers for reading/writing JSON safely: context managers, error handling (Task 2+)."""

from pathlib import Path
from typing import List, Tuple


RAW_DIR = Path("data/raw")                          # where the raw corpus lives
PROCESSED_DIR = Path("data/processed")              # where the index gets saved
DATASETS_DIR = Path("data/datasets")                # where question JSONs live
OUTPUT_DIR = Path("data/output")                    # where results get written

SUPPORTED_EXTENSIONS = {".py", ".md"}              # only chunk these file types


def load_corpus(corpus_dir: str = str(RAW_DIR)) -> List[Tuple[str, str]]:
    """
    Recursively read all .py and .md files from *corpus_dir*.

    Returns a list of (relative_path, file_content) tuples.
    """
    
    results = []
    corpus_dir = Path(corpus_dir)
    for item in corpus_dir.rglob("*"):
        if item.is_file():
            if not (item.suffix in SUPPORTED_EXTENSIONS):
                continue
            try:
                content = item.read_text(encoding="utf-8")
                results.append((str(item.relative_to(corpus_dir)) , content))


            except UnicodeDecodeError:
                pass



    return results
