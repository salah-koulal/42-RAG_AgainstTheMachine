"""Splits Python source files into indexable chunks ."""

from typing import List

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from src.models import MinimalSource


def chunk_python(
    file_path: str, content: str, max_chunk_size: int = 2000
) -> List[MinimalSource]:
    """chunks using Python-specific separator"""
    results: List[MinimalSource] = []

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=max_chunk_size,
        chunk_overlap=0,
        add_start_index=True,
    )

    docs = splitter.create_documents([content])

    for doc in docs:
        start_idx = doc.metadata["start_index"]
        end_idx = start_idx + len(doc.page_content)

        results.append(
            MinimalSource(
                file_path=file_path,
                first_character_index=start_idx,
                last_character_index=end_idx,
            )
        )
    return results
