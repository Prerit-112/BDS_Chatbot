# BDS Abacus: Advanced RAG Chatbot

BDS Abacus is a professional Retrieval-Augmented Generation (RAG) system designed for Bharat Soka Gakkai (BSG) data management and support. It leverages OpenAI's GPT models and ChromaDB for high-performance document-grounded conversations.

## ✨ Key Features

- **Document Grounding**: Answers are strictly based on indexed PDFs and a central context document (`BSG_BDS_Context_Document.md`).
- **Smart Retrieval**: Uses semantic search with ChromaDB and cross-encoder reranking for high precision.
- **Confidence Scoring**: Real-time confidence metrics (High/Medium/Low) for every response.
- **Performance Insights**: Detailed logging of token usage and response timing.
- **Guardrails**: Integrated safety checks for input moderation and output sanity.
- **Modern Streaming UI**: Premium, BSG-themed chat interface with real-time response streaming.
- **Response Caching**: Persistent caching using `diskcache` to reduce latency and API costs.
- **Feedback Loop**: Integrated user feedback mechanism (👍/👎) to continuously improve knowledge base quality.
- **Detailed Logging**: All interactions and feedback are logged to the `logs/` directory.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- OpenAI API Key

### Installation

1. **Create a virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Configure environment:**
   - Copy `.env.example` to `.env`
   - Add your `OPENAI_API_KEY`.

### Data Ingestion

Place your PDFs in the `rag_data/` directory and run:
```powershell
python scripts/ingest_rag.py --reset
```

### Running the App

```powershell
streamlit run streamlit_app.py
```

## 🛠️ Architecture

- **Vector Store**: ChromaDB (Persistent)
- **Embeddings**: `sentence-transformers/all-mpnet-base-v2`
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM**: OpenAI GPT-4o-mini
- **Frontend**: Streamlit

## 📝 Logging & Feedback

- **Application Logs**: `logs/app.log` (JSON format)
- **User Feedback**: `logs/feedback.jsonl`
- **Cache**: `.cache/` directory

---
© 2026 BDS Abacus Team
