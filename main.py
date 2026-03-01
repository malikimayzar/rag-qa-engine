import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

from pipeline.parser import parse_pdf, save_parsed, load_config
from pipeline.embedder import embed_chunks, save_embeddings
from pipeline.faiss_store import build_index, save_index, load_index
from chunking.chunker import chunk_pages, save_chunks
from pipeline.retrieval import hybrid_search
from pipeline.reranker import rerank
from pipeline.groq_llm import answer, answer_stream
from logger import get_logger

logger = get_logger("api")
config = load_config()
indexes = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    index_dir = Path("data/index")
    if index_dir.exists():
        for meta_file in index_dir.glob("*_meta.json"):
            doc_id = meta_file.stem.replace("_meta", "")
            try:
                index, chunks = load_index(doc_id)
                indexes[doc_id] = {"index": index, "chunks": chunks}
                logger.info(f"Loaded index: {doc_id}")
            except Exception as e:
                logger.error(f"Failed to load {doc_id}: {e}")
    yield

app = FastAPI(title="RAG QA Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    doc_id: Optional[str] = None   # None = query all documents
    top_k: int = 5
    stream: bool = False

@app.get("/api/documents")
def list_documents():
    return {"documents": list(indexes.keys())}

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    if doc_id not in indexes:
        raise HTTPException(status_code=404, detail="Document not found")
    # Remove from memory
    del indexes[doc_id]
    # Remove files from disk
    index_dir = Path("data/index")
    for pattern in [f"{doc_id}.faiss", f"{doc_id}_meta.json", f"{doc_id}_embeddings.npy"]:
        f = index_dir / pattern
        if f.exists():
            f.unlink()
    logger.info(f"Deleted: {doc_id}")
    return {"deleted": doc_id}

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    save_path = Path("data/raw") / file.filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    logger.info(f"Uploaded: {file.filename}")
    pages = parse_pdf(str(save_path), config)
    save_parsed(pages)
    chunks = chunk_pages(pages, config)
    save_chunks(chunks)
    embeddings = embed_chunks(chunks)
    doc_id = Path(file.filename).stem
    save_embeddings(embeddings, doc_id)
    index = build_index(embeddings)
    save_index(index, chunks, doc_id)
    indexes[doc_id] = {"index": index, "chunks": chunks}
    return {"doc_id": doc_id, "pages": len(pages), "chunks": len(chunks)}

@app.post("/api/query")
def query_document(req: QueryRequest):
    # Multi-doc: gather chunks from all indexes or specific one
    if req.doc_id:
        if req.doc_id not in indexes:
            raise HTTPException(status_code=404, detail="Document not found")
        target_indexes = {req.doc_id: indexes[req.doc_id]}
    else:
        target_indexes = indexes

    if not target_indexes:
        raise HTTPException(status_code=404, detail="No documents loaded")

    # Retrieve from each index and merge
    all_chunks = []
    for doc_id, idx in target_indexes.items():
        retrieved = hybrid_search(req.query, idx["index"], idx["chunks"], top_k=req.top_k)
        all_chunks.extend(retrieved)

    # Rerank merged results
    reranked = rerank(req.query, all_chunks, top_n=3)

    # Streaming response
    if req.stream:
        def event_stream():
            sources_str = ""
            for chunk_text in answer_stream(req.query, reranked):
                if chunk_text.startswith("__SOURCES__:"):
                    sources_str = chunk_text.replace("__SOURCES__:", "")
                    data = json.dumps({"type": "sources", "content": sources_str})
                else:
                    data = json.dumps({"type": "token", "content": chunk_text})
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Normal response
    result = answer(req.query, reranked)
    return {
        "query": req.query,
        "answer": result["answer"],
        "sources": result["sources"],
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"]
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "documents_loaded": len(indexes)}

# ── Eval endpoints ─────────────────────────────────────────────────────────────
from eval.metrics import run_eval, save_eval

DEFAULT_QUERIES = [
    "apa isi dokumen ini?",
    "berikan ringkasan dokumen",
    "apa poin penting dari dokumen ini?",
    "siapa yang terlibat dalam dokumen ini?",
    "apa kesimpulan dari dokumen ini?",
]

class EvalRequest(BaseModel):
    doc_id: str
    queries: Optional[list[str]] = None

@app.post("/api/eval/run")
def run_evaluation(req: EvalRequest):
    if req.doc_id not in indexes:
        raise HTTPException(status_code=404, detail="Document not found")
    idx = indexes[req.doc_id]
    queries = req.queries or DEFAULT_QUERIES
    results = run_eval(queries, idx["index"], idx["chunks"])
    save_eval(results)
    return results

@app.get("/api/eval/results")
def get_eval_results():
    path = Path("eval/results.json")
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)