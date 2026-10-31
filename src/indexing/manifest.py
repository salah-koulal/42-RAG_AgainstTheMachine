"""File manifest helpers for change detection (Bonus Part)."""

import hashlib
import json
from typing import Dict

from src.io_utils import PROCESSED_DIR

MANIFEST_FILE = PROCESSED_DIR / "file_manifest.json"


def compute_file_hash(content: str) -> str:
    """Return MD5 hash of the file content to detect modifications."""

    return hashlib.md5(content.encode("utf-8")).hexdigest()


def load_manifest() -> Dict[str, str]:
    """Load and return the saved file manifest dictionary."""
    if not MANIFEST_FILE.is_file():
        return {}
    try:
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return dict(data) if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_manifest(manifest: Dict[str, str]) -> None:
    """Persist the file manifest as JSON."""
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
