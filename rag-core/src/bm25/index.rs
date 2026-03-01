use super::scorer::BM25;

pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty() && w.len() > 2)
        .collect()
}

pub fn build_index(texts: &[String]) -> BM25 {
    let corpus: Vec<Vec<String>> = texts.iter().map(|t| tokenize(t)).collect();
    BM25::new(&corpus, 1.5, 0.75)
}
