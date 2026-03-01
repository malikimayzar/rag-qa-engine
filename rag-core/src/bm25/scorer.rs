use std::collections::HashMap;

pub struct BM25 {
    k1: f64,
    b: f64,
    avgdl: f64,
    doc_freqs: Vec<HashMap<String, usize>>,
    idf: HashMap<String, f64>,
    doc_len: Vec<usize>,
    corpus_size: usize,
}

impl BM25 {
    pub fn new(corpus: &[Vec<String>], k1: f64, b: f64) -> Self {
        let corpus_size = corpus.len();
        let doc_freqs: Vec<HashMap<String, usize>> = corpus
            .iter()
            .map(|doc| {
                let mut freq = HashMap::new();
                for word in doc {
                    *freq.entry(word.clone()).or_insert(0) += 1;
                }
                freq
            })
            .collect();

        let doc_len: Vec<usize> = corpus.iter().map(|d| d.len()).collect();
        let avgdl = doc_len.iter().sum::<usize>() as f64 / corpus_size as f64;

        // Hitung IDF
        let mut df: HashMap<String, usize> = HashMap::new();
        for freq_map in &doc_freqs {
            for word in freq_map.keys() {
                *df.entry(word.clone()).or_insert(0) += 1;
            }
        }

        let idf = df
            .into_iter()
            .map(|(word, freq)| {
                let idf_val = ((corpus_size as f64 - freq as f64 + 0.5)
                    / (freq as f64 + 0.5)
                    + 1.0)
                    .ln();
                (word, idf_val)
            })
            .collect();

        BM25 { k1, b, avgdl, doc_freqs, idf, doc_len, corpus_size }
    }

    pub fn score(&self, query: &[String], doc_idx: usize) -> f64 {
        let mut score = 0.0;
        let dl = self.doc_len[doc_idx] as f64;
        let freq_map = &self.doc_freqs[doc_idx];

        for term in query {
            if let Some(&idf) = self.idf.get(term) {
                let tf = *freq_map.get(term).unwrap_or(&0) as f64;
                let numerator = tf * (self.k1 + 1.0);
                let denominator = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl);
                score += idf * numerator / denominator;
            }
        }
        score
    }

    pub fn get_top_k(&self, query: &[String], k: usize) -> Vec<(usize, f64)> {
        let mut scores: Vec<(usize, f64)> = (0..self.corpus_size)
            .map(|i| (i, self.score(query, i)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(k);
        scores
    }
}
