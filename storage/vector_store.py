from pinecone import Pinecone
from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

pc = Pinecone(api_key=settings.pinecone_api_key)


def get_index():
    return pc.Index(settings.pinecone_index_name)


def query_dense(
    vector: list[float],
    tenant_id: str,
    top_k: int = 20,
    filter: dict = None,
) -> list[dict]:
    """Dense vector search. Returns list of { id, score, metadata }."""
    index = get_index()
    f = {"tenant_id": {"$eq": tenant_id}}
    if filter:
        f.update(filter)
    results = index.query(
        vector=vector,
        top_k=top_k,
        namespace=tenant_id,
        filter=f,
        include_metadata=True,
    )
    return [
        {"id": m.id, "score": m.score, "metadata": m.metadata}
        for m in results.matches
    ]


def fetch_by_ids(ids: list[str], tenant_id: str) -> list[dict]:
    """Fetch specific chunks by id."""
    index = get_index()
    results = index.fetch(ids=ids, namespace=tenant_id)
    chunks = []
    for id_, vec in results.vectors.items():
        chunks.append({"id": id_, "metadata": vec.metadata})
    return chunks


def fetch_by_section(section_id: str, tenant_id: str) -> list[dict]:
    """Fetch all chunks belonging to a section — for cross-modal expansion."""
    index = get_index()
    results = index.query(
        vector=[0.0] * settings.openai_embedding_dim,
        filter={
            "tenant_id": {"$eq": tenant_id},
            "section_id": {"$eq": section_id},
        },
        top_k=50,
        namespace=tenant_id,
        include_metadata=True,
    )
    return [
        {"id": m.id, "score": m.score, "metadata": m.metadata}
        for m in results.matches
    ]


def delete_by_document(document_id: str, tenant_id: str):
    """Delete all vectors for a document."""
    index = get_index()
    results = index.query(
        vector=[0.0] * settings.openai_embedding_dim,
        filter={"document_id": {"$eq": document_id}},
        top_k=10000,
        namespace=tenant_id,
        include_metadata=False,
    )
    ids = [m.id for m in results.matches]
    if ids:
        index.delete(ids=ids, namespace=tenant_id)
        logger.info(f"Deleted {len(ids)} vectors for doc: {document_id}")