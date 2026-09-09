import fitz  
import pdfplumber
import base64
import hashlib
import io
import os
from pathlib import Path
from openai import OpenAI
from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)
client = OpenAI(api_key=settings.openai_api_key)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hash(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _image_to_base64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")


# ── Image understanding via GPT-4o Vision ────────────────────────────────────

def describe_image(img_bytes: bytes) -> dict:
    """Send image to GPT-4o, get caption + description + ocr_text back."""
    try:
        b64 = _image_to_base64(img_bytes)
        response = client.chat.completions.create(
            model=settings.openai_llm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Analyze this image from a document. Return a JSON with exactly these keys:\n"
                                "- caption: one short sentence describing the image\n"
                                "- description: detailed structured description including all numbers, "
                                "labels, axes, legends, and data points visible\n"
                                "- ocr_text: all text visible in the image verbatim\n"
                                "Return only valid JSON, no markdown."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )
        import json
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"GPT-4o vision failed: {e}")
        return {"caption": "Image", "description": "", "ocr_text": ""}


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text_per_page(pdf_path: str) -> list[dict]:
    """
    Returns list of:
    { page: int, text: str, section_hint: str }
    """
    results = []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        if not text:
            continue
        results.append({
            "page": page_num,
            "text": text,
            "section_hint": "",   # filled by chunker
        })
    doc.close()
    return results


# ── Table extraction ──────────────────────────────────────────────────────────

def extract_tables(pdf_path: str) -> list[dict]:
    """
    Returns list of:
    { page: int, markdown: str, row_count: int, summary: str }
    """
    results = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    if not table or not table[0]:
                        continue
                    # convert to markdown
                    headers = table[0]
                    rows = table[1:]
                    md_lines = []
                    md_lines.append("| " + " | ".join(str(h or "") for h in headers) + " |")
                    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                    for row in rows:
                        md_lines.append("| " + " | ".join(str(c or "") for c in row) + " |")
                    markdown = "\n".join(md_lines)
                    summary = f"Table on page {page_num + 1}. Columns: {', '.join(str(h or '') for h in headers)}. Rows: {len(rows)}."
                    results.append({
                        "page": page_num,
                        "markdown": markdown,
                        "row_count": len(rows),
                        "summary": summary,
                        "headers": [str(h or "") for h in headers],
                    })
    except Exception as e:
        logger.warning(f"Table extraction failed: {e}")
    return results


# ── Image extraction ──────────────────────────────────────────────────────────

def extract_images(pdf_path: str, output_dir: str, document_id: str) -> list[dict]:
    """
    Extracts images from PDF, saves to disk, runs GPT-4o vision.
    Returns list of:
    { page, image_path, bbox, caption, description, ocr_text, content_hash }
    """
    results = []
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                ext = base_image["ext"]

                # skip tiny images (icons/artifacts)
                if len(img_bytes) < 5000:
                    continue

                # save to disk
                img_filename = f"{document_id}_p{page_num}_i{img_index}.{ext}"
                img_path = os.path.join(output_dir, img_filename)
                with open(img_path, "wb") as f:
                    f.write(img_bytes)

                # get bounding box
                rects = page.get_image_rects(xref)
                bbox = list(rects[0]) if rects else None

                # GPT-4o vision
                logger.info(f"Running GPT-4o vision on page {page_num} image {img_index}")
                vision_result = describe_image(img_bytes)

                results.append({
                    "page": page_num,
                    "image_path": img_path,
                    "bbox": bbox,
                    "caption": vision_result.get("caption", ""),
                    "description": vision_result.get("description", ""),
                    "ocr_text": vision_result.get("ocr_text", ""),
                    "content_hash": _hash(img_bytes),
                    "mime_type": f"image/{ext}",
                })
            except Exception as e:
                logger.warning(f"Failed to extract image xref={xref} page={page_num}: {e}")
                continue

    doc.close()
    return results


# ── Master extractor ──────────────────────────────────────────────────────────

def extract_document(pdf_path: str, document_id: str, data_dir: str) -> dict:
    """
    Main entry point. Extracts text, tables, images from a PDF.
    Returns:
    {
        page_count: int,
        text_pages: [...],
        tables: [...],
        images: [...],
    }
    """
    logger.info(f"Extracting document: {pdf_path}")
    images_dir = os.path.join(data_dir, "images")

    doc = fitz.open(pdf_path)
    page_count = len(doc)
    doc.close()

    text_pages = extract_text_per_page(pdf_path)
    logger.info(f"Extracted text from {len(text_pages)} pages")

    tables = extract_tables(pdf_path)
    logger.info(f"Extracted {len(tables)} tables")

    images = extract_images(pdf_path, images_dir, document_id)
    logger.info(f"Extracted {len(images)} images")

    return {
        "page_count": page_count,
        "text_pages": text_pages,
        "tables": tables,
        "images": images,
    }