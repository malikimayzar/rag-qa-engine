import json
import faiss
import numpy as np
from pathlib import Path
from logger import get_logger

logger = get_logger("faiss_store")

def build_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product (cosine karena udah normalized)
    index.add(embeddings.astype(np.float32))
    logger.info(f"Index built: {index.ntotal} vectors, dim={dim}")
    return index

def save_index(index: faiss.Index, chunks: list[dict], doc_id: str, output_dir: str = "data/index"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Simpan FAISS index
    index_path = Path(output_dir) / f"{doc_id}.faiss"
    faiss.write_index(index, str(index_path))
    
    # Simpan metadata chunks
    meta_path = Path(output_dir) / f"{doc_id}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Index saved: {index_path}")
    return str(index_path)

def load_index(doc_id: str, index_dir: str = "data/index"):
    index_path = Path(index_dir) / f"{doc_id}.faiss"
    meta_path = Path(index_dir) / f"{doc_id}_meta.json"
    
    index = faiss.read_index(str(index_path))
    with open(meta_path, encoding="utf-8") as f:
        chunks = json.load(f)
    
    logger.info(f"Index loaded: {index.ntotal} vectors")
    return index, chunks

def search(index: faiss.Index, chunks: list[dict], query_embedding: np.ndarray, top_k: int = 5):
    query = query_embedding.astype(np.float32).reshape(1, -1)
    scores, indices = index.search(query, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            **chunks[idx],
            "score_dense": float(score)
        })
    return results
