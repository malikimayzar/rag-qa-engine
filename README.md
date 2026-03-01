# 🔍 RAG QA Engine

> **Retrieval-Augmented Generation** system for document question answering — built with a hybrid retrieval pipeline, Rust-powered BM25, and a modern React UI.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![Rust](https://img.shields.io/badge/Rust-1.93-orange?style=flat-square&logo=rust)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?style=flat-square&logo=fastapi)
![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react)
![Groq](https://img.shields.io/badge/LLM-Groq%20%2F%20LLaMA3.3--70B-f55036?style=flat-square)

---

## ✨ Features

- 📄 **PDF Ingestion** — PyMuPDF + pdfplumber fallback, auto-detect slide vs document
- 🔀 **Hybrid Retrieval** — Dense (FAISS) + BM25 (Rust binary) dengan weighted fusion
- 🧠 **Reranking** — Cross-encoder `ms-marco-MiniLM-L-6-v2` untuk presisi lebih tinggi
- 💬 **LLM Synthesis** — Groq API dengan `llama-3.3-70b-versatile`, streaming response
- 📊 **Eval Dashboard** — Faithfulness, context relevance, latency breakdown per query
- 🌐 **Web UI** — React + Vite, split-view, multi-doc query, chat history

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
┌─────────────┐
│   Parser    │  PyMuPDF + pdfplumber
│  + Cleaner  │  header/footer removal
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Chunker   │  auto-detect: slide (1 chunk/page)
│             │  or document (semantic sliding window)
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────┐
│         Hybrid Retrieval         │
│                                  │
│  Dense (FAISS IndexFlatIP)  0.6  │
│  + BM25 (Rust binary)       0.4  │
│  → weighted score fusion         │
└──────┬───────────────────────────┘
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
   Answer + Sources
```

---

## 📁 Project Structure

```
rag-qa-engine/
├── pipeline/
│   ├── parser.py        # PDF parsing (PyMuPDF + pdfplumber)
│   ├── cleaner.py       # Text cleaning pipeline
│   ├── embedder.py      # Sentence embedding (MiniLM)
│   ├── faiss_store.py   # FAISS index build & search
│   ├── retrieval.py     # Hybrid search (dense + BM25)
│   ├── reranker.py      # Cross-encoder reranking
│   └── groq_llm.py      # Groq API + streaming
├── chunking/
│   └── chunker.py       # Auto-detect slide vs document chunking
├── rag-core/            # Rust crate — BM25 search binary
│   └── src/
│       ├── bin/bm25_search.rs
│       └── bm25/{mod,scorer,index}.rs
├── eval/
│   ├── metrics.py       # Faithfulness, relevance, latency metrics
│   ├── runner.py        # Eval runner
│   └── results.json     # Latest eval results
├── frontend/            # React + Vite UI
│   └── src/
│       ├── App.jsx
│       ├── App.css
│       └── EvalDashboard.jsx
├── bridge/              # PyO3 bindings (placeholder)
├── data/                # Runtime data (git-ignored)
│   ├── raw/             # Uploaded PDFs
│   ├── processed/       # Parsed + chunked JSON
│   └── index/           # FAISS indexes
├── main.py              # FastAPI application
├── logger.py            # Logging setup
├── config.yaml          # All configuration
├── pyproject.toml       # Python dependencies (uv)
└── .env                 # API keys (git-ignored)
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [Rust](https://rustup.rs/) (untuk build BM25 binary)
- Node.js 18+ (untuk frontend)
- Groq API key → [console.groq.com](https://console.groq.com)

### 1. Clone & Setup

```bash
git clone https://github.com/<username>/rag-qa-engine.git
cd rag-qa-engine

# Install Python dependencies
uv sync

# Build Rust BM25 binary
cd rag-core && cargo build --release && cd ..
```

### 2. Environment

```bash
cp .env.example .env
# Edit .env dan isi GROQ_API_KEY
```

### 3. Run Backend

```bash
source .venv/bin/activate
uv run uvicorn main:app --reload --port 8000
```

### 4. Run Frontend

```bash
cd frontend
npm install
npm run dev
# Buka http://localhost:5173
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check + loaded docs count |
| `GET` | `/api/documents` | List indexed documents |
| `POST` | `/api/upload` | Upload & index PDF |
| `DELETE` | `/api/documents/{doc_id}` | Delete document from index |
| `POST` | `/api/query` | Query (single/multi-doc, streaming) |
| `POST` | `/api/eval/run` | Run evaluation on a document |
| `GET` | `/api/eval/results` | Get latest eval results |

### Query Request

```json
{
  "query": "apa syarat menjadi GSA?",
  "doc_id": "my-document",   
  "top_k": 5,
  "stream": true
}
```

> `doc_id` opsional — jika tidak diisi, query akan dijalankan ke semua dokumen yang ter-index.

---

## 📊 Eval Results

Diukur pada dokumen GSA (13 chunks):

| Query | Faithfulness | Relevance | Latency |
|-------|-------------|-----------|---------|
| apa syarat menjadi GSA? | 61.3% | 41.7% | 1054ms |
| apa manfaat ikut program GSA? | 62.9% | 40.0% | 560ms |
| bagaimana tips self interview GSA? | 69.6% | 40.0% | 1029ms |
| apa itu Google Student Ambassador? | 70.7% | 33.3% | 847ms |
| siapa saja speaker di acara ini? | 66.7% | 5.6% | 847ms |
| **Average** | **66.2%** | **32.1%** | **~860ms** |

> Latency diukur setelah model embedding warm-up (first query lebih lambat karena load model).

---

## ⚙️ Configuration

Semua konfigurasi ada di `config.yaml`:

```yaml
retrieval:
  dense_weight: 0.6   # bobot FAISS
  bm25_weight: 0.4    # bobot Rust BM25
  top_k: 5

reranking:
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_n: 3

llm:
  model: llama-3.3-70b-versatile
  max_tokens: 1024
  temperature: 0.1
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Ingestion** | PyMuPDF, pdfplumber |
| **Chunking** | Custom semantic chunker (Python) |
| **Embedding** | `paraphrase-multilingual-MiniLM-L12-v2` |
| **Dense Index** | FAISS `IndexFlatIP` |
| **Sparse Search** | BM25 — Rust binary via subprocess |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **LLM** | Groq API — `llama-3.3-70b-versatile` |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | React 19 + Vite 7 |
| **Package Manager** | uv (Python), npm (JS) |

---

## 🗺️ Roadmap

- [x] PDF ingestion + cleaning
- [x] Semantic chunking (auto-detect slide/document)
- [x] Hybrid retrieval (FAISS + Rust BM25)
- [x] Cross-encoder reranking
- [x] Groq LLM integration with streaming
- [x] FastAPI backend
- [x] React UI — split view, chat history, multi-doc
- [x] Eval dashboard (faithfulness, relevance, latency)
- [ ] PyO3 Rust-Python bindings (bridge/)
- [ ] Export chat to PDF
- [ ] Feedback per answer (👍👎)
- [ ] Docker deployment

---

## 📝 License

MIT License — feel free to use, modify, and distribute.

---

<p align="center">Built with ❤️ as a portfolio project</p>