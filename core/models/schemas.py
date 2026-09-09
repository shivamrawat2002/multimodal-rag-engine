from pydantic import BaseModel, Field
from datetime import datetime

class DocumentUploadResponse(BaseModel):
    document_id: str
    status: str
    message: str

class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    page_count: int
    chunk_count: int
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

class ChunkMetadata(BaseModel):
    tenant_id: str
    document_id: str
    document_version: int = 1
    chunk_id: str
    parent_chunk_id: str | None = None
    type: str                        # text | table | image | doc_summary | section_summary
    content: str
    caption: str | None = None
    description: str | None = None
    ocr_text: str | None = None
    page_start: int = 0
    page_end: int = 0
    bbox: list[float] | None = None
    section_id: str | None = None
    section_title: str | None = None
    source: str = ""
    storage_uri: str | None = None
    mime_type: str | None = None
    content_hash: str = ""
    created_at: str = ""
    access_control: list[str] = Field(default_factory=list)

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"
    tenant_id: str = "tenant_dev"
    user_id: str = "user_dev"
    document_ids: list[str] | None = None

class Citation(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    page: int
    type: str
    bbox: list[float] | None = None

class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    session_id: str
    intent: str
    modalities_used: list[str]
    retrieval_latency_ms: int
    llm_latency_ms: int
    total_latency_ms: int

class HealthResponse(BaseModel):
    status: str
    env: str
    services: dict