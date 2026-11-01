from fastapi import FastAPI
from pydantic import BaseModel, Field, StringConstraints
from typing import Annotated

from src.models import MinimalSource
from src.generation.generator import QwenGenerator
from src.retrieval.bm25_retriever import BM25Retriever


app = FastAPI(title="RAG against the machine API", version="0.1.0")
retriever = BM25Retriever()
generator = QwenGenerator()


@app.get("/health")
def check_status() -> dict[str, str]:
    return {"status": "ok"}


class SearchRequest(BaseModel):
    query: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1)
    ]
    k: int = Field(default=5, ge=1, le=50)


class SearchResponse(BaseModel):
    query: str
    results: list[MinimalSource]


class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: list[MinimalSource]


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    results = retriever.retrieve(query=request.query, k=request.k)
    return SearchResponse(query=request.query, results=results)


@app.post("/answer", response_model=AnswerResponse)
def answer(request: SearchRequest) -> AnswerResponse:
    sources = retriever.retrieve(query=request.query, k=request.k)
    generated_answer = generator.generate_answer(query=request.query, sources=sources)
    return AnswerResponse(query=request.query, answer=generated_answer, sources=sources)
