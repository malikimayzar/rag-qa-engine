pub mod ingestion;
pub mod chunking;
pub mod bm25;

use pyo3::prelude::*;
use crate::bm25::index::{build_index, tokenize};

// ─── PyO3 Result struct ───────────────────────────────────────────────────────
#[pyclass]
#[derive(Clone)]
pub struct BM25Result {
    #[pyo3(get)]
    pub index: usize,
    #[pyo3(get)]
    pub score: f64,
}

#[pymethods]
impl BM25Result {
    fn __repr__(&self) -> String {
        format!("BM25Result(index={}, score={:.4})", self.index, self.score)
    }
}

// ─── PyO3 BM25Searcher ────────────────────────────────────────────────────────
/// BM25 searcher yang bisa dipakai langsung dari Python.
///
/// Contoh:
/// ```python
/// from rag_core import BM25Searcher
/// searcher = BM25Searcher(["teks pertama", "teks kedua"])
/// results = searcher.search("query saya", top_k=5)
/// for r in results:
///     print(r.index, r.score)
/// ```
#[pyclass]
pub struct BM25Searcher {
    bm25: crate::bm25::scorer::BM25,
    doc_count: usize,
}

#[pymethods]
impl BM25Searcher {
    /// Buat BM25Searcher dari list teks
    #[new]
    pub fn new(texts: Vec<String>) -> Self {
        let doc_count = texts.len();
        let bm25 = build_index(&texts);
        BM25Searcher { bm25, doc_count }
    }

    /// Search query, return top_k hasil sorted by score desc
    pub fn search(&self, query: &str, top_k: usize) -> Vec<BM25Result> {
        let query_tokens = tokenize(query);
        let mut results = self.bm25.get_top_k(&query_tokens, top_k);
        results.retain(|(_, score)| *score > 0.0);
        results
            .into_iter()
            .map(|(index, score)| BM25Result { index, score })
            .collect()
    }

    /// Jumlah dokumen di index
    pub fn doc_count(&self) -> usize {
        self.doc_count
    }
}

// ─── Module registration ──────────────────────────────────────────────────────
#[pymodule]
fn rag_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BM25Searcher>()?;
    m.add_class::<BM25Result>()?;
    Ok(())
}