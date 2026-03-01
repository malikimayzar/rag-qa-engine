use std::io::{self, Read};
use serde::{Deserialize, Serialize};
use rag_core::bm25::index::{build_index, tokenize};

#[derive(Deserialize)]
struct Input {
    query: String,
    texts: Vec<String>,
    top_k: usize,
}

#[derive(Serialize)]
struct Result {
    index: usize,
    score: f64,
}

fn main() {
    let mut input_str = String::new();
    io::stdin().read_to_string(&mut input_str).unwrap();

    let input: Input = serde_json::from_str(&input_str).unwrap();
    let bm25 = build_index(&input.texts);
    let query_tokens = tokenize(&input.query);
    let results = bm25.get_top_k(&query_tokens, input.top_k);

    let output: Vec<Result> = results
        .into_iter()
        .map(|(i, s)| Result { index: i, score: s })
        .collect();

    println!("{}", serde_json::to_string(&output).unwrap());
}
