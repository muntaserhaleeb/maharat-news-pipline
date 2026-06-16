from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ConfigSaveRequest(BaseModel):
    content: str
    note: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    collection: str = "primary"
    limit: int = 10
    score_threshold: float = 0.0
    category: Optional[str] = None
    year: Optional[int] = None
    quarter: Optional[str] = None
    chunk_type: Optional[str] = None
    language: Optional[str] = None
