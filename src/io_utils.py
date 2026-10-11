"""Shared helpers for reading and writing files safely."""

from pathlib import Path
from typing import List, Tuple

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
DATASETS_DIR = Path("data/datasets")
OUTPUT_DIR = Path("data/output")

SUPPORTED_EXTENSIONS = {".py", ".md", ".txt"}


def load_corpus(corpus_dir: str = str(RAW_DIR)) -> List[Tuple[str, str]]:
    """Recursively read all supported files from corpus_dir.

    Returns a list of (relative_path, file_content) tuples.
    """
    results: List[Tuple[str, str]] = []
    files = Path(corpus_dir)
    if not files.exists():
        return results

    for item in sorted(files.rglob("*")):
        if item.is_file() and item.suffix in SUPPORTED_EXTENSIONS:
            try:
                content = item.read_text(encoding="utf-8", errors="ignore")
                results.append((str(item.relative_to(files)), content))
            except Exception:
                pass
    return results
