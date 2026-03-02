import os
import json
import time
import re
import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, field_validator, Field
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

from pipeline.parser import parse_pdf, save_parsed, load_config
from pipeline.embedder import embed_chunks, save_embeddings, warmup as warmup_embedder
from pipeline.faiss_store import build_index, save_index, load_index
from chunking.chunker import chunk_pages, save_chunks
from pipeline.retrieval import hybrid_search
from pipeline.reranker import rerank, warmup as warmup_reranker
from pipeline.groq_llm import answer, answer_stream
from logger import get_logger

logger = get_logger("api")
config = load_config()
indexes = {}
_executor = ThreadPoolExecutor(max_workers=2)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost,http://localhost:5173").split(",")
MAX_UPLOAD_SIZE  = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20")) * 1024 * 1024
MAX_DOCS         = int(os.getenv("MAX_DOCS", "20"))
RATE_LIMIT_RPM   = int(os.getenv("RATE_LIMIT_RPM", "30"))

_rate_store: dict = defaultdict(list)

def rate_limit(request: Request):
    ip = request.client.host
    now = time.time()
    _rate_store[ip] = [t for t in _rate_store[ip] if now - t < 60.0]
    if len(_rate_store[ip]) >= RATE_LIMIT_RPM:
        raise HTTPException(status_code=429, detail=f"Too many requests. Max {RATE_LIMIT_RPM} req/min.")
    _rate_store[ip].append(now)

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

    # Warmup models di background thread
    import threading
    threading.Thread(target=warmup_embedder, daemon=True).start()
    threading.Thread(target=warmup_reranker, daemon=True).start()
    yield
    _executor.shutdown(wait=False)

app = FastAPI(title="RAG QA Engine", version="0.1.0", docs_url=None, redoc_url=None, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)

@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

SAFE_DOC_ID = re.compile(r'^[\w\-. ]{1,100}$')

def validate_doc_id(doc_id: str) -> str:
    if not SAFE_DOC_ID.match(doc_id):
        raise HTTPException(status_code=400, detail="Invalid doc_id format")
    return doc_id

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    doc_id: Optional[str] = Field(None, max_length=100)
    top_k: int = Field(5, ge=1, le=20)
    stream: bool = False

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v):
        v = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', v).strip()
        if not v:
            raise ValueError("Query cannot be empty")
        return v

    @field_validator("doc_id")
    @classmethod
    def validate_doc(cls, v):
        if v is not None and not SAFE_DOC_ID.match(v):
            raise ValueError("Invalid doc_id format")
        return v

class EvalRequest(BaseModel):
    doc_id: str = Field(..., max_length=100)
    queries: Optional[list[str]] = Field(None, max_length=20)

@app.get("/api/health")
def health():
    return {"status": "ok", "documents_loaded": len(indexes)}

@app.get("/api/documents", dependencies=[Depends(rate_limit)])
def list_documents():
    return {"documents": list(indexes.keys())}

@app.delete("/api/documents/{doc_id}", dependencies=[Depends(rate_limit)])
def delete_document(doc_id: str):
    doc_id = validate_doc_id(doc_id)
    if doc_id not in indexes:
        raise HTTPException(status_code=404, detail="Document not found")
    del indexes[doc_id]
    index_dir = Path("data/index")
    for pattern in [f"{doc_id}.faiss", f"{doc_id}_meta.json", f"{doc_id}_embeddings.npy"]:
        f = index_dir / pattern
        if f.exists():
            f.unlink()
    logger.info(f"Deleted: {doc_id}")
    return {"deleted": doc_id}

@app.post("/api/upload", dependencies=[Depends(rate_limit)])
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    safe_name = re.sub(r'[^\w\-. ]', '_', Path(file.filename).stem)[:80]
    safe_filename = safe_name + ".pdf"

    if len(indexes) >= MAX_DOCS:
        raise HTTPException(status_code=400, detail=f"Max {MAX_DOCS} documents allowed")

    content = await file.read(MAX_UPLOAD_SIZE + 1)
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_SIZE//1024//1024}MB")

    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Invalid PDF file")

    save_path = Path("data/raw") / safe_filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(content)

    logger.info(f"Uploaded: {safe_filename} ({len(content)//1024}KB)")

    try:
        def process():
            pages = parse_pdf(str(save_path), config)
            save_parsed(pages)
            chunks = chunk_pages(pages, config)
            save_chunks(chunks)
            embeddings = embed_chunks(chunks)
            doc_id = safe_name
            save_embeddings(embeddings, doc_id)
            index = build_index(embeddings)
            save_index(index, chunks, doc_id)
            return pages, chunks, doc_id, index

        loop = asyncio.get_event_loop()
        pages, chunks, doc_id, index = await loop.run_in_executor(_executor, process)
        indexes[doc_id] = {"index": index, "chunks": chunks}
        return {"doc_id": doc_id, "pages": len(pages), "chunks": len(chunks)}
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process PDF")

@app.post("/api/query", dependencies=[Depends(rate_limit)])
def query_document(req: QueryRequest):
    if req.doc_id:
        validate_doc_id(req.doc_id)
        if req.doc_id not in indexes:
            raise HTTPException(status_code=404, detail="Document not found")
        target_indexes = {req.doc_id: indexes[req.doc_id]}
    else:
        target_indexes = indexes

    if not target_indexes:
        raise HTTPException(status_code=404, detail="No documents loaded")

    all_chunks = []
    for doc_id, idx in target_indexes.items():
        retrieved = hybrid_search(req.query, idx["index"], idx["chunks"], top_k=req.top_k)
        all_chunks.extend(retrieved)

    reranked = rerank(req.query, all_chunks, top_n=3)

    if req.stream:
        def event_stream():
            try:
                for chunk_text in answer_stream(req.query, reranked):
                    if chunk_text.startswith("__SOURCES__:"):
                        sources_str = chunk_text.replace("__SOURCES__:", "")
                        data = json.dumps({"type": "sources", "content": sources_str})
                    else:
                        data = json.dumps({"type": "token", "content": chunk_text})
                    yield f"data: {data}\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': 'Stream error'})}\n\n"
            finally:
                yield "data: [DONE]\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        result = answer(req.query, reranked)
        return {
            "query": req.query,
            "answer": result["answer"],
            "sources": result["sources"],
            "tokens_in": result["tokens_in"],
            "tokens_out": result["tokens_out"]
        }
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate answer")

from eval.metrics import run_eval, save_eval

DEFAULT_QUERIES = [
    "apa isi dokumen ini?",
    "berikan ringkasan dokumen",
    "apa poin penting dari dokumen ini?",
    "siapa yang terlibat dalam dokumen ini?",
    "apa kesimpulan dari dokumen ini?",
]

@app.post("/api/eval/run", dependencies=[Depends(rate_limit)])
def run_evaluation(req: EvalRequest):
    validate_doc_id(req.doc_id)
    if req.doc_id not in indexes:
        raise HTTPException(status_code=404, detail="Document not found")
    idx = indexes[req.doc_id]
    queries = req.queries or DEFAULT_QUERIES
    try:
        results = run_eval(queries, idx["index"], idx["chunks"])
        save_eval(results)
        return results
    except Exception as e:
        logger.error(f"Eval error: {e}")
        raise HTTPException(status_code=500, detail="Eval failed")

@app.get("/api/eval/results")
def get_eval_results():
    path = Path("eval/results.json")
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)
