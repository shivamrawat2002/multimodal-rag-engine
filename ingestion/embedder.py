import time
from openai import OpenAI
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.models.schemas import ChunkMetadata

settings = get_settings()
logger = get_logger(__name__)
client = OpenAI(api_key=settings.openai_api_key)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _batch(lst: list, size: int):
    """Yield successive batches from list."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def _embed_texts(texts: list[str], retries: int = 3) -> list[list[float]]:
    """
    Call OpenAI embeddings API with retry on rate limit.
    Returns list of embedding vectors.
    """
    for attempt in range(retries):
        try:
            response = client.embeddings.create(
                model=settings.openai_embedding_model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"Embedding attempt {attempt + 1} failed: {e} — retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Embedding failed after {retries} attempts")


# ── Main embedder ─────────────────────────────────────────────────────────────

def embed_chunks(chunks: list[ChunkMetadata], batch_size: int = 100) -> list[dict]:
    """
    Embeds all chunks in batches.
    Returns list of dicts ready for Pinecone upsert:
    {
        id: chunk_id,
        values: [float, ...],
        metadata: { all chunk fields }
    }
    """
    logger.info(f"Embedding {len(chunks)} chunks in batches of {batch_size}")
    results = []

    for batch_num, batch in enumerate(_batch(chunks, batch_size)):
        # what we embed: content field (description for images, summary for tables, text for text)
        texts = [c.content for c in batch]

        logger.info(f"Embedding batch {batch_num + 1} ({len(texts)} chunks)")
        vectors = _embed_texts(texts)

        for chunk, vector in zip(batch, vectors):
            # build metadata dict — pinecone stores this alongside the vector
            metadata = {
                "tenant_id": chunk.tenant_id,
                "document_id": chunk.document_id,
                "document_version": chunk.document_version,
                "chunk_id": chunk.chunk_id,
                "type": chunk.type,
                "content": chunk.content[:1000],   # pinecone metadata limit
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "source": chunk.source,
                "created_at": chunk.created_at,
            }

            # optional fields — only add if present
            if chunk.parent_chunk_id:
                metadata["parent_chunk_id"] = chunk.parent_chunk_id
            if chunk.section_id:
                metadata["section_id"] = chunk.section_id
            if chunk.section_title:
                metadata["section_title"] = chunk.section_title
            if chunk.caption:
                metadata["caption"] = chunk.caption
            if chunk.description:
                metadata["description"] = chunk.description[:500]
            if chunk.ocr_text:
                metadata["ocr_text"] = chunk.ocr_text[:500]
            if chunk.bbox:
                metadata["bbox"] = chunk.bbox
            if chunk.storage_uri:
                metadata["storage_uri"] = chunk.storage_uri
            if chunk.mime_type:
                metadata["mime_type"] = chunk.mime_type
            if chunk.content_hash:
                metadata["content_hash"] = chunk.content_hash
            if chunk.access_control:
                metadata["access_control"] = chunk.access_control

            results.append({
                "id": chunk.chunk_id,
                "values": vector,
                "metadata": metadata,
            })

        # small sleep between batches to avoid rate limits
        if batch_num > 0:
            time.sleep(0.5)

    logger.info(f"Embedding done: {len(results)} vectors ready")
    return results