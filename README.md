# 🔍 RAG QA Engine

> A production-ready **Retrieval-Augmented Generation (RAG)** system for question answering over PDF documents — featuring hybrid search, a Rust-powered BM25 engine via PyO3, cross-encoder reranking, and a modern React UI.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![Rust](https://img.shields.io/badge/Rust-1.93-orange?style=flat-square&logo=rust)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?style=flat-square&logo=fastapi)
![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)
![Groq](https://img.shields.io/badge/LLM-Groq%20%2F%20LLaMA3.3--70B-f55036?style=flat-square)

---

## ✨ Features

- 📄 **PDF Ingestion** — PyMuPDF + pdfplumber fallback, auto-detects slide decks vs long-form documents
- 🔀 **Hybrid Retrieval** — Dense vector search (FAISS) + sparse BM25 (Rust via PyO3) with weighted score fusion
- 🦀 **Rust BM25 Engine** — Native BM25 compiled as a Python extension module via PyO3 — no subprocess overhead
- 🧠 **Cross-Encoder Reranking** — `ms-marco-MiniLM-L-6-v2` for precision-focused re-scoring
- 💬 **LLM Synthesis** — Groq API with `llama-3.3-70b-versatile`, streaming response via SSE
- 📊 **Evaluation Framework** — Faithfulness, context relevance, and latency metrics per query
- 🌐 **Web UI** — React + Vite with split-view layout, multi-doc query, and chat history
- 🐳 **Docker Ready** — Full stack deployable with a single `docker compose up`

---

## 🏗️ Architecture

\`\`\`
PDF Upload
    │
    ▼
┌─────────────┐
│   Parser    │  PyMuPDF + pdfplumber fallback
│  + Cleaner  │  header/footer removal, dedup
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Chunker   │  auto-detect: slide   → 1 chunk/page
│             │              document → semantic sliding window
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────┐
│           Hybrid Retrieval           │
│                                      │
│  Dense  FAISS IndexFlatIP    × 0.6   │
│  Sparse BM25 (Rust / PyO3)   × 0.4   │
│  → weighted score fusion             │
└──────┬───────────────────────────────┘
       │
       ▼
┌─────────────┐
│  Reranker   │  cross-encoder/ms-marco-MiniLM-L-6-v2
│  (top_n=3)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Groq LLM   │  llama-3.3-70b-versatile
│  Streaming  │  SSE streaming response
└──────┬──────┘
       │
       ▼
   Answer + Cited Sources
\`\`\`

---

## 📁 Project Structure

\`\`\`
rag-qa-engine/
├── pipeline/
│   ├── parser.py        # PDF parsing (PyMuPDF + pdfplumber)
│   ├── cleaner.py       # Text cleaning & deduplication
│   ├── embedder.py      # Sentence embedding (MiniLM)
│   ├── faiss_store.py   # FAISS index build & search
│   ├── retrieval.py     # Hybrid search (dense + BM25)
│   ├── reranker.py      # Cross-encoder reranking
│   └── groq_llm.py      # Groq API + SSE streaming
├── chunking/
│   └── chunker.py       # Auto-detect slide vs document chunking
├── rag-core/            # Rust crate — BM25 as PyO3 extension
│   └── src/
│       ├── lib.rs                    # PyO3 module: BM25Searcher class
│       ├── bin/bm25_search.rs        # Standalone CLI binary
│       └── bm25/{mod,scorer,index}.rs
├── bridge/
│   └── bindings.py      # Python wrapper with pure-Python BM25 fallback
├── eval/
│   ├── metrics.py       # Faithfulness, relevance, latency
│   ├── runner.py        # Eval runner
│   └── results.json     # Latest eval output
├── frontend/            # React + Vite UI
├── docker/
│   └── nginx.conf
├── data/                # Runtime data (git-ignored)
├── main.py              # FastAPI application
├── config.yaml          # Centralized configuration
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
└── pyproject.toml
\`\`\`

---

## ⚡ Quick Start

### Option A — Docker (Recommended)

\`\`\`bash
git clone https://github.com/<username>/rag-qa-engine.git
cd rag-qa-engine

cp .env.example .env
# Edit .env → add GROQ_API_KEY

docker compose up -d
\`\`\`

| Service      | URL                        |
|--------------|----------------------------|
| Frontend     | http://localhost           |
| Backend API  | https://rag-qa-engine-production.up.railway.app      |
| Swagger Docs | https://rag-qa-engine-production.up.railway.app/docs |

### Option B — Local Development

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), [Rust](https://rustup.rs/), Node.js 18+, Groq API key

\`\`\`bash
git clone https://github.com/<username>/rag-qa-engine.git
cd rag-qa-engine
uv sync
cd rag-core && maturin develop --release && cd ..
cp .env.example .env
\`\`\`

\`\`\`bash
# Backend
uv run uvicorn main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev
\`\`\`

**.env.example**
\`\`\`env
GROQ_API_KEY=your_groq_api_key_here
\`\`\`

---

## 🔌 API Reference

| Method   | Endpoint                  | Description                         |
|----------|---------------------------|-------------------------------------|
| GET      | /api/health               | Health check + loaded doc count     |
| GET      | /api/documents            | List all indexed documents          |
| POST     | /api/upload               | Upload and auto-index a PDF         |
| DELETE   | /api/documents/{doc_id}   | Remove a document from index        |
| POST     | /api/query                | Ask a question (supports streaming) |
| POST     | /api/eval/run             | Run evaluation on a document        |
| GET      | /api/eval/results         | Retrieve latest eval results        |

\`\`\`json
{
  "query": "What are the requirements to become a GSA?",
  "doc_id": "my-document",
  "top_k": 5,
  "stream": true
}
\`\`\`

> doc_id is optional — omit it to query across all indexed documents.

---

## 📊 Evaluation Results

Measured on a 13-chunk GSA event document:

| Query | Faithfulness | Relevance | Latency |
|-------|:---:|:---:|:---:|
| What are the requirements to become a GSA? | 61.3% | 41.7% | 1054ms |
| What are the benefits of the GSA program? | 62.9% | 40.0% | 560ms |
| What are tips for the GSA self-interview? | 69.6% | 40.0% | 1029ms |
| What is Google Student Ambassador? | 70.7% | 33.3% | 847ms |
| Who are the speakers at this event? | 66.7% | 5.6% | 847ms |
| **Average** | **66.2%** | **32.1%** | **~868ms** |

> Latency measured after model warm-up. First query is slower due to embedding model loading (~26s).

---

## ⚙️ Configuration

\`\`\`yaml
retrieval:
  dense_weight: 0.6
  bm25_weight: 0.4
  top_k: 5

reranking:
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_n: 3

llm:
  model: llama-3.3-70b-versatile
  max_tokens: 1024
  temperature: 0.1
\`\`\`

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Ingestion** | PyMuPDF, pdfplumber |
| **Chunking** | Custom semantic chunker |
| **Embedding** | paraphrase-multilingual-MiniLM-L12-v2 (CPU) |
| **Dense Index** | FAISS IndexFlatIP |
| **Sparse Search** | BM25 — Rust extension via PyO3 |
| **Reranking** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **LLM** | Groq API — llama-3.3-70b-versatile |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | React 19 + Vite 7 + nginx |
| **Containerization** | Docker Compose |
| **Package Manager** | uv (Python), npm (JS) |

---

## 🗺️ Roadmap

- [x] PDF ingestion + cleaning pipeline
- [x] Semantic chunking with auto-detect (slide / document)
- [x] Hybrid retrieval — FAISS + Rust BM25 (PyO3)
- [x] Cross-encoder reranking
- [x] Groq LLM integration with SSE streaming
- [x] FastAPI backend with Swagger docs
- [x] React UI — split view, chat history, multi-doc query
- [x] Evaluation framework (faithfulness, relevance, latency)
- [x] Docker Compose full-stack deployment
- [ ] Cloud deployment (Railway / Render)
- [ ] Per-answer feedback (👍 👎)
- [ ] Export chat history to PDF
- [ ] Document management UI (delete, detail view)

---

## 📝 License

MIT License — free to use, modify, and distribute.

---

<p align="center">Built with ❤️ as a portfolio project</p>
