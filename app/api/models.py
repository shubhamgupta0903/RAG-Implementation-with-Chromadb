from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str


class DocumentMetadata(BaseModel):
    filename: str
    upload_date: datetime
    processing_status: str

    class Config:
        from_attributes = True