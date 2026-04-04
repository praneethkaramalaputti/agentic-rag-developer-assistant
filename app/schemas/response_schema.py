from pydantic import BaseModel
from typing import List

class MatchedChunk(BaseModel):
    source: str
    page_number: int
    text: str

class QueryResponse(BaseModel):
    query: str
    matched_chunks: List[MatchedChunk]