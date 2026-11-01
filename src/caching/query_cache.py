"""Disk-based query cache for search and answer results (Bonus Part)."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.io_utils import PROCESSED_DIR

CACHE_FILE = Path("cache/query_cache.json")


def _get_key(mode: str, query: str, k: int) -> str:
    """Normalize mode, query, and k into a deterministic key."""
    # Rebuilt indexes must not reuse results pointing at previous chunk ranges.
    files = [PROCESSED_DIR / "bm25_index" / "params.index.json",
             PROCESSED_DIR / "chunks_metadata.pkl"]
    revision = []
    for path in files:
        try:
            stat = path.stat()
            revision.append(f"{stat.st_mtime_ns}:{stat.st_size}")
        except OSError:
            revision.append("missing")
    return f"{'|'.join(revision)}:{mode}:{query.strip().lower()}:{k}"


def get_cached(mode: str, query: str, k: int) -> Optional[Any]:
    """Return cached result if present on disk, otherwise None."""
    if not CACHE_FILE.exists():
        return None

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        return data.get(_get_key(mode, query, k))
    except Exception:
        return None


def set_cached(mode: str, query: str, k: int, result: Any) -> None:
    """Save a query result into the disk cache JSON file."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

    data[_get_key(mode, query, k)] = result
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def clear_cache() -> None:
    """Delete or clear the query cache file."""
    if CACHE_FILE.exists():
        try:
            CACHE_FILE.unlink()
        except Exception:
            pass
