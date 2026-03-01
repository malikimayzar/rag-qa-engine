import json
import re
from pathlib import Path
from logger import get_logger

logger = get_logger("chunker")

def split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def chunk_slide(page: dict) -> list[dict]:
    return [{
        "doc_id": page["doc_id"],
        "chunk_id": f"{page['doc_id']}_p{page['page']}_c0",
        "page": page["page"],
        "section": page["section"],
        "text": page["text"],
        "chunk_type": "slide",
        "token_estimate": len(page["text"].split())
    }]

def chunk_document(page: dict, min_size: int = 100, max_size: int = 512) -> list[dict]:
    sentences = split_sentences(page["text"])
    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_len = 0
    chunk_idx = 0

    for sent in sentences:
        sent_len = len(sent.split())

        if current_len + sent_len > max_size and current_len >= min_size:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "doc_id": page["doc_id"],
                "chunk_id": f"{page['doc_id']}_p{page['page']}_c{chunk_idx}",
                "page": page["page"],
                "section": page["section"],
                "text": chunk_text,
                "chunk_type": "document",
                "token_estimate": current_len
            })
            current_chunk = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
            current_len = sum(len(s.split()) for s in current_chunk)
            chunk_idx += 1

        current_chunk.append(sent)
        current_len += sent_len

    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text.split()) >= 10:
            chunks.append({
                "doc_id": page["doc_id"],
                "chunk_id": f"{page['doc_id']}_p{page['page']}_c{chunk_idx}",
                "page": page["page"],
                "section": page["section"],
                "text": chunk_text,
                "chunk_type": "document",
                "token_estimate": len(chunk_text.split())
            })

    return chunks

def chunk_pages(pages: list[dict], config: dict) -> list[dict]:
    min_size = config["chunking"]["min_chunk_size"]
    max_size = config["chunking"]["max_chunk_size"]

    all_chunks = []
    for page in pages:
        doc_type = page.get("doc_type", "document")
        if doc_type == "slide":
            chunks = chunk_slide(page)
        else:
            chunks = chunk_document(page, min_size, max_size)
        all_chunks.extend(chunks)

    logger.info(f"Total chunks: {len(all_chunks)} dari {len(pages)} halaman")
    return all_chunks

def save_chunks(chunks: list[dict], output_dir: str = "data/processed") -> str:
    if not chunks:
        return ""

    doc_id = chunks[0]["doc_id"]
    output_path = Path(output_dir) / f"{doc_id}_chunks.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved chunks: {output_path}")
    return str(output_path)

def print_stats(chunks: list[dict]):
    sizes = [c["token_estimate"] for c in chunks]
    print(f"Total chunks  : {len(chunks)}")
    print(f"Rata-rata size: {sum(sizes)/len(sizes):.1f} tokens")
    print(f"Min size      : {min(sizes)} tokens")
    print(f"Max size      : {max(sizes)} tokens")
    
    types = {}
    for c in chunks:
        t = c["chunk_type"]
        types[t] = types.get(t, 0) + 1
    print(f"Tipe chunk    : {types}")
