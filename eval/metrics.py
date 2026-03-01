import time
import json
from pathlib import Path
from logger import get_logger

logger = get_logger("eval")

def evaluate_pipeline(query: str, chunks_retrieved: list[dict], answer_result: dict, latencies: dict) -> dict:
    """Evaluasi satu query."""
    
    # Context relevance — seberapa relevan chunks ke query
    query_words = set(query.lower().split())
    relevance_scores = []
    for chunk in chunks_retrieved:
        chunk_words = set(chunk["text"].lower().split())
        overlap = len(query_words & chunk_words) / max(len(query_words), 1)
        relevance_scores.append(overlap)
    avg_relevance = sum(relevance_scores) / max(len(relevance_scores), 1)

    # Faithfulness — apakah jawaban pakai kata dari konteks
    answer_words = set(answer_result["answer"].lower().split())
    context_words = set()
    for chunk in chunks_retrieved:
        context_words.update(chunk["text"].lower().split())
    
    if answer_words:
        faithfulness = len(answer_words & context_words) / len(answer_words)
    else:
        faithfulness = 0.0

    # Token efficiency
    tokens_in = answer_result.get("tokens_in", 0)
    tokens_out = answer_result.get("tokens_out", 0)

    return {
        "query": query,
        "context_relevance": round(avg_relevance, 4),
        "faithfulness": round(faithfulness, 4),
        "chunks_used": len(chunks_retrieved),
        "latency_retrieval_ms": round(latencies.get("retrieval", 0), 1),
        "latency_rerank_ms": round(latencies.get("rerank", 0), 1),
        "latency_llm_ms": round(latencies.get("llm", 0), 1),
        "latency_total_ms": round(sum(latencies.values()), 1),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "answer_preview": answer_result["answer"][:100]
    }

def run_eval(queries: list[str], index, chunks: list[dict]) -> list[dict]:
    """Run eval untuk semua queries."""
    from pipeline.retrieval import hybrid_search
    from pipeline.reranker import rerank
    from pipeline.groq_llm import answer

    results = []
    for query in queries:
        logger.info(f"Evaluating: {query[:50]}")

        # Retrieval
        t0 = time.time()
        retrieved = hybrid_search(query, index, chunks, top_k=5)
        t_retrieval = (time.time() - t0) * 1000

        # Rerank
        t0 = time.time()
        reranked = rerank(query, retrieved, top_n=3)
        t_rerank = (time.time() - t0) * 1000

        # LLM
        t0 = time.time()
        ans = answer(query, reranked)
        t_llm = (time.time() - t0) * 1000

        metric = evaluate_pipeline(
            query=query,
            chunks_retrieved=reranked,
            answer_result=ans,
            latencies={
                "retrieval": t_retrieval,
                "rerank": t_rerank,
                "llm": t_llm
            }
        )
        results.append(metric)

    return results

def save_eval(results: list[dict], output_path: str = "eval/results.json"):
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Eval saved: {output_path}")

def print_eval_table(results: list[dict]):
    print(f"\n{'='*80}")
    print(f"{'QUERY':<35} {'FAITH':>6} {'REL':>6} {'TOT_MS':>8} {'TOK_IN':>7}")
    print(f"{'='*80}")
    for r in results:
        print(f"{r['query'][:34]:<35} {r['faithfulness']:>6.3f} {r['context_relevance']:>6.3f} {r['latency_total_ms']:>8.1f} {r['tokens_in']:>7}")
    print(f"{'='*80}")
    
    avg_faith = sum(r['faithfulness'] for r in results) / len(results)
    avg_rel = sum(r['context_relevance'] for r in results) / len(results)
    avg_lat = sum(r['latency_total_ms'] for r in results) / len(results)
    print(f"{'AVERAGE':<35} {avg_faith:>6.3f} {avg_rel:>6.3f} {avg_lat:>8.1f}")
