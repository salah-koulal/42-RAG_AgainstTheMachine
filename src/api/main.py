from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.models import MinimalSource
app = FastAPI(title="RAG against the machine API", version="0.1.0")


@app.get("/health")
def check_status() -> dict[str, str]:
    return {"status": "ok"}


class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=50)


class SearchResponse(BaseModel):
    query: str
    results: list[MinimalSource]
