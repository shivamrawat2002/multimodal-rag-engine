from pinecone import Pinecone, ServerlessSpec
from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

# ── Init Pinecone ─────────────────────────────────────────────────────────────

pc = Pinecone(api_key=settings.pinecone_api_key)


def get_or_create_index():
    """Create index if it doesn't exist, return index object."""
    existing = [i.name for i in pc.list_indexes()]

    if settings.pinecone_index_name not in existing:
        logger.info(f"Creating Pinecone index: {settings.pinecone_index_name}")
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.openai_embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
        logger.info("Index created")
    else:
        logger.info(f"Index already exists: {settings.pinecone_index_name}")

    return pc.Index(settings.pinecone_index_name)


# ── Upsert ────────────────────────────────────────────────────────────────────

def upsert_chunks(vectors: list[dict], tenant_id: str, batch_size: int = 100):
    """
    Upsert embedded chunks into Pinecone.
    Uses tenant_id as namespace for isolation.
    vectors: list of { id, values, metadata }
    """
    index = get_or_create_index()

    logger.info(f"Upserting {len(vectors)} vectors to namespace: {tenant_id}")

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch, namespace=tenant_id)
        logger.info(f"Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")

    logger.info("Upsert complete")


# ── Delete ────────────────────────────────────────────────────────────────────

def delete_document(document_id: str, tenant_id: str):
    """
    Delete all vectors for a document from Pinecone.
    Called when a document is deleted.
    """
    index = get_or_create_index()

    logger.info(f"Deleting vectors for document: {document_id}")

    # fetch chunk ids by metadata filter then delete
    results = index.query(
        vector=[0.0] * settings.openai_embedding_dim,
        filter={"document_id": {"$eq": document_id}},
        top_k=10000,
        namespace=tenant_id,
        include_metadata=False,
    )

    ids_to_delete = [m.id for m in results.matches]

    if ids_to_delete:
        index.delete(ids=ids_to_delete, namespace=tenant_id)
        logger.info(f"Deleted {len(ids_to_delete)} vectors for document: {document_id}")
    else:
        logger.info("No vectors found to delete")


# ── Stats ─────────────────────────────────────────────────────────────────────

def index_stats(tenant_id: str) -> dict:
    """Return index stats for a tenant namespace."""
    index = get_or_create_index()
    stats = index.describe_index_stats()
    ns = stats.namespaces.get(tenant_id, {})
    return {
        "total_vectors": stats.total_vector_count,
        "tenant_vectors": getattr(ns, "vector_count", 0),
        "dimension": stats.dimension,
    }