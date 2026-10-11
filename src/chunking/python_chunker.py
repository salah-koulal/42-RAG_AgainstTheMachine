"""Splits Python source files into indexable chunks."""
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from src.models import MinimalSource
from src.io_utils import load_corpus


def chunk_python(file_path: str, content: str, max_chunk_size: int = 2000) -> List[MinimalSource]:
    results = []

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=max_chunk_size,
        chunk_overlap = 0,
        add_start_index = True
    )

    docs = splitter.create_documents([content])

    for doc in docs:
        start_idx = doc.metadata["start_index"]
        end_idx = start_idx + len(doc.page_content)

        results.append(
            MinimalSource(
                file_path=file_path,
                first_character_index=start_idx,
                last_character_index=end_idx
            )
        )
    return results

if __name__ == "__main__":
    files = load_corpus()
    chunks = chunk_python(files[0][0], files[0][1])
    print(files[0][0])
    for c in chunks:
        print(f"start index: {c.first_character_index} | end_index: {c.last_character_index} | size: {c.last_character_index - c.first_character_index}")