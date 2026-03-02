import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from logger import get_logger

logger = get_logger("embedder")
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model: SentenceTransformer | None = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Model loaded!")
    return _model

def warmup():
    """Pre-load model saat startup biar first query tidak lambat."""
    model = get_model()
    model.encode(["warmup"], show_progress_bar=False)
    logger.info("Embedding model warmed up!")

def embed_chunks(chunks: list[dict], batch_size: int = 32) -> np.ndarray:
    model = get_model()
    texts = [c["text"] for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    logger.info(f"Embedding shape: {embeddings.shape}")
    return embeddings

def save_embeddings(embeddings: np.ndarray, doc_id: str, output_dir: str = "data/index"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"{doc_id}_embeddings.npy"
    np.save(path, embeddings)
    logger.info(f"Saved embeddings: {path}")
    return str(path)

def load_embeddings(doc_id: str, index_dir: str = "data/index") -> np.ndarray:
    path = Path(index_dir) / f"{doc_id}_embeddings.npy"
    return np.load(path)
