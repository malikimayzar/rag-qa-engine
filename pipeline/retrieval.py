import json
import subprocess
import numpy as np
from pathlib import Path
from logger import get_logger
from pipeline.faiss_store import load_index, search
from pipeline.embedder import get_model

logger = get_logger("retrieval")

BM25_BINARY = "./rag-core/target/release/bm25_search"

def bm25_search(query: str, chunks: list[dict], top_k: int = 10) -> list[dict]:
    texts = [c["text"] for c in chunks]
    input_data = json.dumps({"query": query, "texts": texts, "top_k": top_k})

    result = subprocess.run(
        [BM25_BINARY],
        input=input_data,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        logger.error(f"BM25 error: {result.stderr}")
        return []

    raw = json.loads(result.stdout)
    results = []
    for r in raw:
        chunk = chunks[r["index"]].copy()
        chunk["score_bm25"] = r["score"]
        chunk["score_dense"] = 0.0
        results.append(chunk)
    return results

def dense_search(query: str, index, chunks: list[dict], top_k: int = 10) -> list[dict]:
    model = get_model()
    q_emb = model.encode([query], normalize_embeddings=True)
    results = search(index, chunks, q_emb, top_k=top_k)
    for r in results:
        r["score_bm25"] = 0.0
    return results

def normalize_scores(results: list[dict], score_key: str) -> list[dict]:
    scores = [r[score_key] for r in results]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return results
    for r in results:
        r[score_key] = (r[score_key] - min_s) / (max_s - min_s)
    return results

def hybrid_search(
    query: str,
    index,
    chunks: list[dict],
    top_k: int = 5,
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4
) -> list[dict]:
    fetch_k = min(top_k * 3, len(chunks))

    # Ambil dari kedua retriever
    dense_results = dense_search(query, index, chunks, top_k=fetch_k)
    bm25_results = bm25_search(query, chunks, top_k=fetch_k)

    # Normalize
    if dense_results:
        dense_results = normalize_scores(dense_results, "score_dense")
    if bm25_results:
        bm25_results = normalize_scores(bm25_results, "score_bm25")

    # Gabungkan by chunk_id
    merged = {}
    for r in dense_results:
        cid = r["chunk_id"]
        merged[cid] = r.copy()
        merged[cid]["score_hybrid"] = dense_weight * r["score_dense"]

    for r in bm25_results:
        cid = r["chunk_id"]
        if cid in merged:
            merged[cid]["score_bm25"] = r["score_bm25"]
            merged[cid]["score_hybrid"] += bm25_weight * r["score_bm25"]
        else:
            r["score_hybrid"] = bm25_weight * r["score_bm25"]
            merged[cid] = r

    # Sort by hybrid score
    results = sorted(merged.values(), key=lambda x: x["score_hybrid"], reverse=True)
    results = results[:top_k]

    logger.info(f"Hybrid search '{query}': {len(results)} results")
    return results
