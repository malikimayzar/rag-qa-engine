import fitz
import pdfplumber
import json
import yaml
import re
from pathlib import Path
from logger import get_logger
from pipeline.cleaner import clean_slide_text, clean_document_text

logger = get_logger("parser")

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def detect_doc_type(pages_raw: list[str]) -> str:
    """Deteksi otomatis jenis dokumen: slide atau document."""
    avg_len = sum(len(p) for p in pages_raw) / max(len(pages_raw), 1)
    return "slide" if avg_len < 500 else "document"

def parse_with_pdfplumber(pdf_path: str) -> list[dict]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append({
                "page": page_num,
                "section": "Unknown",
                "text": text.strip()
            })
    return pages

def parse_with_pymupdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        sections = []
        paragraphs = []
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                spans = line["spans"]
                text = " ".join([s["text"] for s in spans]).strip()
                if not text:
                    continue
                avg_size = sum(s["size"] for s in spans) / len(spans)
                if avg_size > 13:
                    sections.append(text)
                else:
                    paragraphs.append(text)
        pages.append({
            "page": page_num,
            "section": sections[-1] if sections else "Unknown",
            "text": " ".join(paragraphs) if paragraphs else " ".join(sections)
        })
    doc.close()
    return pages

def parse_pdf(pdf_path: str, config: dict) -> list[dict]:
    path = Path(pdf_path)
    doc_id = path.stem
    logger.info(f"Parsing: {path.name}")

    try:
        raw_pages = parse_with_pdfplumber(pdf_path)
    except Exception as e:
        logger.warning(f"pdfplumber gagal: {e}, fallback ke pymupdf")
        raw_pages = parse_with_pymupdf(pdf_path)

    # Auto-detect jenis dokumen
    raw_texts = [p["text"] for p in raw_pages]
    doc_type = detect_doc_type(raw_texts)
    logger.info(f"Tipe dokumen terdeteksi: {doc_type}")

    results = []
    for page in raw_pages:
        if doc_type == "slide":
            text = clean_slide_text(page["text"])
        else:
            text = clean_document_text(page["text"])

        if len(text) < 10:
            logger.debug(f"Skip halaman {page['page']}: terlalu pendek")
            continue

        results.append({
            "doc_id": doc_id,
            "page": page["page"],
            "section": page["section"],
            "text": text,
            "doc_type": doc_type
        })

    logger.info(f"Selesai: {len(results)} halaman valid")
    return results

def save_parsed(results: list[dict], output_dir: str = "data/processed") -> str:
    if not results:
        logger.warning("Tidak ada hasil untuk disimpan")
        return ""

    doc_id = results[0]["doc_id"]
    output_path = Path(output_dir) / f"{doc_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved: {output_path}")
    return str(output_path)
