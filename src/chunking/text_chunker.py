"""Splits Markdown/text files into indexable chunks (Task 1)."""
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.models import MinimalSource
from src.io_utils import load_corpus

def chunk_text(file_path: str, content: str, max_chunk_size: int = 2000) -> List[MinimalSource]:
    results = []
    
    # Initialize the Langchain splitter
    # add_start_index=True is CRITICAL here, otherwise you won't know where the chunk started!
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=0,
        add_start_index=True 
    )
    
    # Langchain creates "Document" objects
    docs = splitter.create_documents([content])
    
    # We must convert Langchain's Documents into your MinimalSource Pydantic models
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


# def chunk_text(file_path: str, content: str, max_chunk_size: int = 2000) -> List[MinimalSource]:
#     """
#     Split a Markdown/text file into chunks based on blank-line boundaries.

#     Each chunk respects *max_chunk_size* (in characters). If a paragraph is
#     larger than max_chunk_size, it is further split by newlines ("\n"), and
#     finally by brute-force slicing if still too big.

#     Returns a list of MinimalSource objects with correct character indices.

#     How character tracking works:
#         raw:    "# Title\n\nParagraph one.\n\nSection two."
#         index:   0       8 9             24 25           40
#         chunk1 → first_character_index=0, last_character_index=8
#         chunk2 → first_character_index=9, last_character_index=24
#         chunk3 → first_character_index=25, last_character_index=40

#     Steps:
#         1. Split *content* by SEPARATOR to get raw paragraphs
#         2. For each paragraph, track its start position in the original content
#            (use content.index() or a running cursor to find where it starts)
#         3. If a paragraph exceeds max_chunk_size, split it further (by "\n")
#         4. If a sub-chunk still exceeds max_chunk_size, slice it by size
#         5. For each final piece, create a MinimalSource with:
#                file_path=file_path,
#                first_character_index=start,
#                last_character_index=start + len(piece)
#         6. Skip empty chunks (stripped content == "")
#         7. Return the list of MinimalSource objects
#     """
#     results: List[MinimalSource] = []
#     raw_paragraphs = content.split("\n\n")
#     current_search_idx = 0
#     for para in raw_paragraphs:
#         if not para.strip():
#             continue
#         if len(para) > max_chunk_size:
#             pieces = para.split("\n")
#         else:
#             pieces = [para]
        
#         for piece in pieces:
#             if not piece.strip():
#                 continue
            
#             if len(piece) > max_chunk_size:
#                 sub_pieces = [piece[i: i+ max_chunk_size] for i in range(0, len(piece), max_chunk_size)]
#             else:
#                 sub_pieces = [piece]
            
#             for sub_piece in sub_pieces:
#                 start_idx = content.find(sub_piece, current_search_idx)

#                 if start_idx != -1:
#                     end_idx = start_idx + len(sub_piece)

#                     results.append(
#                         MinimalSource(
#                             file_path=file_path,
#                             first_character_index=start_idx,
#                             last_character_index=end_idx
#                         )
#                     )
#                     current_search_idx = end_idx
#     return results


if __name__ == "__main__":
    content = load_corpus()
    files = load_corpus("data/raw")
    print(f"Found {len(files)} files")
    for path, content in files[2:3]:
        chunks = chunk_text(path, content)

    
    for c in chunks:
        print(f"path:{c.file_path} [{c.first_character_index}:{c.last_character_index}] - Size: {c.last_character_index - c.first_character_index}")
