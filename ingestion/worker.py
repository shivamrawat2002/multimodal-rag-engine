import uuid
import hashlib
import os
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.models.db_models import Document
from ingestion.extractor import extract_document
from ingestion.chunker import chunk_document
from ingestion.embedder import embed_chunks
from ingestion.indexer import upsert_chunks

settings = get_settings()
logger = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _document_id() -> str:
    return "doc_" + str(uuid.uuid4())[:8]


def _file_hash(file_bytes: bytes) -> str:
    return "sha256:" + hashlib.sha256(file_bytes).hexdigest()


async def _update_status(db: AsyncSession, document_id: str, status: str, error: str = None):
    """Update document status in DB."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if doc:
        doc.status = status
        doc.updated_at = datetime.utcnow()
        if error:
            doc.error_message = error
        await db.commit()
        logger.info(f"Document {document_id} → {status}")


# ── Duplicate check ───────────────────────────────────────────────────────────

async def check_duplicate(db: AsyncSession, content_hash: str, tenant_id: str) -> str | None:
    """
    Returns existing document_id if duplicate found, else None.
    """
    result = await db.execute(
        select(Document).where(
            Document.content_hash == content_hash,
            Document.tenant_id == tenant_id,
            Document.status == "READY",
        )
    )
    existing = result.scalar_one_or_none()
    if existing:
        logger.info(f"Duplicate detected — returning existing doc: {existing.id}")
        return existing.id
    return None


# ── Register document in DB ───────────────────────────────────────────────────

async def register_document(
    db: AsyncSession,
    filename: str,
    content_hash: str,
    storage_uri: str,
    tenant_id: str,
    user_id: str,
) -> str:
    """Create document row in DB, return document_id."""
    document_id = _document_id()
    doc = Document(
        id=document_id,
        tenant_id=tenant_id,
        user_id=user_id,
        filename=filename,
        content_hash=content_hash,
        status="UPLOADED",
        storage_uri=storage_uri,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(doc)
    await db.commit()
    logger.info(f"Registered document: {document_id}")
    return document_id


# ── Save uploaded file ────────────────────────────────────────────────────────

def save_upload(file_bytes: bytes, filename: str, document_id: str) -> str:
    """Save PDF to local disk, return storage path."""
    uploads_dir = os.path.join(settings.local_storage_path, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    safe_name = f"{document_id}_{filename}"
    file_path = os.path.join(uploads_dir, safe_name)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    logger.info(f"Saved PDF: {file_path}")
    return file_path


# ── Main ingestion pipeline ───────────────────────────────────────────────────

async def run_ingestion(
    db: AsyncSession,
    document_id: str,
    pdf_path: str,
    filename: str,
    tenant_id: str,
):
    """
    Full pipeline:
    PROCESSING → EXTRACTING → EMBEDDING → INDEXING → READY
    """
    try:
        # EXTRACTING
        await _update_status(db, document_id, "EXTRACTING")
        extracted = extract_document(
            pdf_path=pdf_path,
            document_id=document_id,
            data_dir=settings.local_storage_path,
        )

        # CHUNKING (part of extracting phase)
        chunks = chunk_document(
            extracted=extracted,
            document_id=document_id,
            tenant_id=tenant_id,
            source=filename,
        )

        # EMBEDDING
        await _update_status(db, document_id, "EMBEDDING")
        vectors = embed_chunks(chunks)

        # INDEXING
        await _update_status(db, document_id, "INDEXING")
        upsert_chunks(vectors=vectors, tenant_id=tenant_id)

        # update page + chunk count
        result = await db.execute(select(Document).where(Document.id == document_id))
        doc = result.scalar_one_or_none()
        if doc:
            doc.page_count = extracted["page_count"]
            doc.chunk_count = len(chunks)
            doc.status = "READY"
            doc.updated_at = datetime.utcnow()
            await db.commit()

        logger.info(f"Ingestion complete: {document_id} — {len(chunks)} chunks indexed")

    except Exception as e:
        logger.error(f"Ingestion failed for {document_id}: {e}")
        await _update_status(db, document_id, "FAILED", error=str(e))
        raise