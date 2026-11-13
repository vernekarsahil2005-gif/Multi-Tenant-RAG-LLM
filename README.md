# Secure Multi-Tenant RAG System with Local LLM Service

An enterprise-grade, privacy-first Retrieval-Augmented Generation (RAG) platform designed for isolated multi-tenant document querying. Built with a fully local AI backend to guarantee complete data confidentiality and zero third-party telemetry.

---

## 🛠️ Tech Stack & Architecture

- **Backend Framework:** Python, FastAPI
- **Frontend Interface:** Streamlit
- **AI / LLM Engine:** Ollama (Llama 3 / local models), PyTorch, Sentence Transformers (`all-MiniLM-L6-v2`, `bge-reranker-small`)
- **Retrieval Pipeline:** LangChain, BM25 sparse keyword search + dense vector semantic search, Reciprocal Rank Fusion (RRF)
- **Data Architecture:** Tenant-isolated document partitioning (`client_id` namespace enforcement)

---

## 📂 Project Structure

```text
Multi-Tenant-RAG-LLM/
│
├── app.py                  # Streamlit user interface and interactive chat dashboard
├── rag_core.py             # Core retrieval, embedding generation, and LLM orchestration
├── pipeline.py             # Document ingestion, text chunking, and indexing pipeline
├── evaluate_rag.py         # RAG evaluation and performance metrics suite
├── test_rag.py             # Unit tests for retrieval and context accuracy
├── requirements.txt        # Python dependency manifest
└── documents/              # Isolated tenant data directories
    ├── Client_A/           # Partitioned knowledge base for Client A
    ├── Client_B/           # Partitioned knowledge base for Client B
    └── Client_C/           # Partitioned knowledge base for Client C
