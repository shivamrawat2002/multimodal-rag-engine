import uuid
import tiktoken
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.models.schemas import ChunkMetadata
from datetime import datetime

settings = get_settings()
logger = get_logger(__name__)

enc = tiktoken.get_encoding("cl100k_base")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_id() -> str:
    return "chunk_" + str(uuid.uuid4())[:8]

def _section_id() -> str:
    return "sec_" + str(uuid.uuid4())[:8]

def _token_count(text: str) -> int:
    return len(enc.encode(text))

def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into token-aware chunks with overlap."""
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


# ── Section detection ─────────────────────────────────────────────────────────

def _detect_section(text: str, page: int) -> str:
    """
    Simple heuristic section detection.
    Looks for short lines that look like headings.
    Falls back to 'Page N'.
    """
    lines = text.strip().split("\n")
    for line in lines[:5]:
        line = line.strip()
        if 2 < len(line) < 80 and not line.endswith("."):
            return line
    return f"Page {page + 1}"


# ── Text chunking ─────────────────────────────────────────────────────────────

def chunk_text_pages(
    text_pages: list[dict],
    document_id: str,
    tenant_id: str,
    source: str,
) -> list[ChunkMetadata]:
    """
    Hierarchical chunking for text:
    - doc summary chunk (one per document)
    - section summary chunk (one per section)
    - content chunks (512 tokens / 50 overlap)
    """
    chunks = []
    chunk_size = settings.chunk_size_tokens
    overlap = settings.chunk_overlap_tokens

    # group pages by detected section
    sections: dict[str, dict] = {}
    for page_data in text_pages:
        section_title = _detect_section(page_data["text"], page_data["page"])
        if section_title not in sections:
            sections[section_title] = {
                "section_id": _section_id(),
                "section_title": section_title,
                "pages": [],
                "texts": [],
            }
        sections[section_title]["pages"].append(page_data["page"])
        sections[section_title]["texts"].append(page_data["text"])

    # doc-level summary chunk
    all_text = " ".join(p["text"] for p in text_pages)
    doc_summary = all_text[:1000]  # first 1000 chars as doc summary
    doc_chunk_id = _chunk_id()
    chunks.append(ChunkMetadata(
        tenant_id=tenant_id,
        document_id=document_id,
        chunk_id=doc_chunk_id,
        parent_chunk_id=None,
        type="doc_summary",
        content=doc_summary,
        page_start=0,
        page_end=text_pages[-1]["page"] if text_pages else 0,
        section_id="doc_root",
        section_title="Document Summary",
        source=source,
        created_at=datetime.utcnow().isoformat(),
    ))

    # section + content chunks
    for section_title, section_data in sections.items():
        section_id = section_data["section_id"]
        full_section_text = "\n".join(section_data["texts"])
        page_start = min(section_data["pages"])
        page_end = max(section_data["pages"])

        # section summary chunk
        section_summary = full_section_text[:500]
        sec_chunk_id = _chunk_id()
        chunks.append(ChunkMetadata(
            tenant_id=tenant_id,
            document_id=document_id,
            chunk_id=sec_chunk_id,
            parent_chunk_id=doc_chunk_id,
            type="section_summary",
            content=section_summary,
            page_start=page_start,
            page_end=page_end,
            section_id=section_id,
            section_title=section_title,
            source=source,
            created_at=datetime.utcnow().isoformat(),
        ))

        # content chunks
        text_splits = _split_text(full_section_text, chunk_size, overlap)
        for split in text_splits:
            if not split.strip():
                continue
            chunks.append(ChunkMetadata(
                tenant_id=tenant_id,
                document_id=document_id,
                chunk_id=_chunk_id(),
                parent_chunk_id=sec_chunk_id,
                type="text",
                content=split,
                page_start=page_start,
                page_end=page_end,
                section_id=section_id,
                section_title=section_title,
                source=source,
                created_at=datetime.utcnow().isoformat(),
            ))

    logger.info(f"Text chunking done: {len(chunks)} chunks")
    return chunks


# ── Table chunking ────────────────────────────────────────────────────────────

def chunk_tables(
    tables: list[dict],
    document_id: str,
    tenant_id: str,
    source: str,
) -> list[ChunkMetadata]:
    """
    - Small table (<=20 rows) → single chunk
    - Large table (>20 rows)  → summary chunk + row-group chunks
    """
    chunks = []
    threshold = settings.table_row_threshold

    for table in tables:
        section_id = _section_id()
        page = table["page"]
        markdown = table["markdown"]
        summary = table["summary"]
        row_count = table["row_count"]

        if row_count <= threshold:
            # single chunk — embed the markdown directly
            chunks.append(ChunkMetadata(
                tenant_id=tenant_id,
                document_id=document_id,
                chunk_id=_chunk_id(),
                parent_chunk_id=None,
                type="table",
                content=markdown,
                page_start=page,
                page_end=page,
                section_id=section_id,
                section_title=summary[:80],
                source=source,
                created_at=datetime.utcnow().isoformat(),
            ))
        else:
            # large table — summary chunk first (embed this for retrieval)
            summary_chunk_id = _chunk_id()
            chunks.append(ChunkMetadata(
                tenant_id=tenant_id,
                document_id=document_id,
                chunk_id=summary_chunk_id,
                parent_chunk_id=None,
                type="table",
                content=summary,
                page_start=page,
                page_end=page,
                section_id=section_id,
                section_title=summary[:80],
                source=source,
                created_at=datetime.utcnow().isoformat(),
            ))

            # row group chunks (20 rows each)
            rows = markdown.split("\n")
            header = rows[:2]  # header + separator
            data_rows = rows[2:]
            group_size = 20
            for i in range(0, len(data_rows), group_size):
                group = data_rows[i:i + group_size]
                group_md = "\n".join(header + group)
                chunks.append(ChunkMetadata(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    chunk_id=_chunk_id(),
                    parent_chunk_id=summary_chunk_id,
                    type="table",
                    content=group_md,
                    page_start=page,
                    page_end=page,
                    section_id=section_id,
                    section_title=f"{summary[:60]} [rows {i+1}-{i+len(group)}]",
                    source=source,
                    created_at=datetime.utcnow().isoformat(),
                ))

    logger.info(f"Table chunking done: {len(chunks)} chunks")
    return chunks


# ── Image chunking ────────────────────────────────────────────────────────────

def chunk_images(
    images: list[dict],
    document_id: str,
    tenant_id: str,
    source: str,
) -> list[ChunkMetadata]:
    """One chunk per image — embed the description for retrieval."""
    chunks = []

    for img in images:
        # embed description (richer than caption) — fallback to caption
        embed_content = img.get("description") or img.get("caption") or "Image"

        chunks.append(ChunkMetadata(
            tenant_id=tenant_id,
            document_id=document_id,
            chunk_id=_chunk_id(),
            parent_chunk_id=None,
            type="image",
            content=embed_content,
            caption=img.get("caption"),
            description=img.get("description"),
            ocr_text=img.get("ocr_text"),
            page_start=img["page"],
            page_end=img["page"],
            bbox=img.get("bbox"),
            source=source,
            storage_uri=img.get("image_path"),
            mime_type=img.get("mime_type"),
            content_hash=img.get("content_hash", ""),
            created_at=datetime.utcnow().isoformat(),
        ))

    logger.info(f"Image chunking done: {len(chunks)} chunks")
    return chunks


# ── Master chunker ────────────────────────────────────────────────────────────

def chunk_document(
    extracted: dict,
    document_id: str,
    tenant_id: str,
    source: str,
) -> list[ChunkMetadata]:
    """Runs all chunkers and returns combined list."""
    all_chunks = []
    all_chunks += chunk_text_pages(extracted["text_pages"], document_id, tenant_id, source)
    all_chunks += chunk_tables(extracted["tables"], document_id, tenant_id, source)
    all_chunks += chunk_images(extracted["images"], document_id, tenant_id, source)
    logger.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks