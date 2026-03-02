import time
from sentence_transformers import CrossEncoder
from logger import get_logger

logger = get_logger("reranker")
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model: CrossEncoder | None = None

def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        logger.info(f"Loading reranker: {MODEL_NAME}")
        _model = CrossEncoder(MODEL_NAME, max_length=512)
        logger.info("Reranker loaded!")
    return _model

def warmup():
    """Pre-load reranker saat startup."""
    model = get_reranker()
    model.predict([("warmup query", "warmup text")])
    logger.info("Reranker warmed up!")

def rerank(query: str, results: list[dict], top_n: int = 3) -> list[dict]:
    if not results:
        return []
    model = get_reranker()
    pairs = [(query, r["text"]) for r in results]
    t0 = time.time()
    scores = model.predict(pairs)
    latency = (time.time() - t0) * 1000
    for i, r in enumerate(results):
        r["score_rerank"] = float(scores[i])
    reranked = sorted(results, key=lambda x: x["score_rerank"], reverse=True)[:top_n]
    logger.info(f"Reranked {len(results)} → {len(reranked)} results ({latency:.1f}ms)")
    return reranked
