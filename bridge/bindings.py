"""
bridge/bindings.py
──────────────────
Python wrapper untuk rag_core PyO3 module.
Kalau binary belum di-build, fallback ke pure Python BM25.
"""

import logging
logger = logging.getLogger("bridge")

# Coba import dari Rust binary dulu
try:
    from rag_core import BM25Searcher as _RustBM25Searcher, BM25Result
    RUST_AVAILABLE = True
    logger.info("✅ rag_core Rust binary loaded — PyO3 BM25 active")
except ImportError:
    RUST_AVAILABLE = False
    logger.warning("⚠ rag_core not found — falling back to pure Python BM25")


class BM25Searcher:
    """
    Unified BM25 interface.
    Otomatis pakai Rust jika tersedia, fallback ke Python jika tidak.

    Contoh:
        searcher = BM25Searcher(["teks pertama", "teks kedua", "teks ketiga"])
        results = searcher.search("query saya", top_k=5)
        for r in results:
            print(r["index"], r["score"])
    """

    def __init__(self, texts: list[str]):
        self.texts = texts
        if RUST_AVAILABLE:
            self._searcher = _RustBM25Searcher(texts)
            self._backend = "rust"
        else:
            self._searcher = _PythonBM25(texts)
            self._backend = "python"

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Return list of {"index": int, "score": float}"""
        raw = self._searcher.search(query, top_k)
        if RUST_AVAILABLE:
            return [{"index": r.index, "score": r.score} for r in raw]
        return raw

    def doc_count(self) -> int:
        return len(self.texts)

    @property
    def backend(self) -> str:
        return self._backend


# ── Pure Python BM25 fallback ─────────────────────────────────────────────────
import math
from collections import Counter

class _PythonBM25:
    """Fallback BM25 jika Rust binary belum di-build."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = [self._tokenize(t) for t in texts]
        self.n = len(self.corpus)
        self.avgdl = sum(len(d) for d in self.corpus) / max(self.n, 1)
        self.doc_freqs = [Counter(d) for d in self.corpus]
        self.idf = self._compute_idf()

    def _tokenize(self, text: str) -> list[str]:
        import re
        tokens = re.sub(r"[^\w\s]", " ", text.lower()).split()
        return [t for t in tokens if len(t) > 2]

    def _compute_idf(self) -> dict[str, float]:
        df = Counter()
        for doc in self.corpus:
            df.update(set(doc))
        idf = {}
        for term, freq in df.items():
            idf[term] = math.log((self.n - freq + 0.5) / (freq + 0.5) + 1)
        return idf

    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        score = 0.0
        dl = len(self.corpus[doc_idx])
        freq_map = self.doc_freqs[doc_idx]
        for term in query_tokens:
            if term not in self.idf:
                continue
            tf = freq_map.get(term, 0)
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf[term] * num / den
        return score

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        tokens = self._tokenize(query)
        scores = [(i, self._score(tokens, i)) for i in range(self.n)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [{"index": i, "score": s} for i, s in scores[:top_k] if s > 0]