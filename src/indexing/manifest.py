from textwrap import indent
from pydantic.v1 import JsonError
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from src.models import MinimalSource

MANIFEST_FILE = Path("data/processed/file_manifest.json")
CHUNKS_BY_FILE = Path("data/processed/chunks_metadata.pkl")



def compute_file_hash(content: str) -> str:
    """Return MD5 or SHA256 of the file content to detect if it changed."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()

def detect_changed_files(
    current_files: List[Tuple[str, str]], 
    manifest: Dict[str, str]
) -> Tuple[List[Tuple[str, str]], Set[str]]:
    """Compare current files against manifest.
    
    Returns:
        changed_or_new: files that need to be re-chunked
        deleted_paths: files that were deleted and should be removed
    """
    changed_or_new = []
    current_paths = {file_path for file_path, _ in current_files}

    for file_path, content in current_files:
        current_hash = compute_file_hash(content)

        if file_path not in manifest or manifest[file_path] != current_hash:
            changed_or_new.append((file_path, content))

    deleted_paths = set(manifest.keys()) - current_paths

    return changed_or_new, deleted_paths



def load_manifest() -> Dict[str, str]:
    """Load and return the saved file manifest."""
    if not MANIFEST_FILE.is_file():
        return {}

    try:
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            content = json.load(f)
    except json.JSONDecodeError:
        return {}
    return content


def save_manifest(manifest: Dict[str, str]) -> None:
    """Persist the file manifest as JSON."""
    manifest_dir = Path(MANIFEST_FILE)
    manifest_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE,"w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)




def build_index_incremental(corpus_dir: str = "data/raw") -> None:
    """Incrementally update the index by re-chunking only modified files."""

    

if __name__ == "__main__":
    test_manifest = {
        "example.py": "hash-one",
        "README.md": "hash-two",
    }
    save_manifest(test_manifest)
    print(load_manifest())