from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Document(BaseModel):
    doc_id: str
    filename: str
    content: str
    upload_date: datetime
    page_count: int
    themes: Optional[List[str]] = []
    citations: Optional[List[dict]] = []
    embedding: Optional[List[float]] = [] 