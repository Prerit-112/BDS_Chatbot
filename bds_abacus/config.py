"""Paths and defaults for RAG and chatbot."""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CHROMA_PATH = Path(os.environ.get("RAG_CHROMA_PATH", PROJECT_ROOT / "rag_data" / "chroma_db"))
CONTEXT_DOC_PATH = PROJECT_ROOT / "data" / "BSG_BDS_Context_Document.md"

EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "all-mpnet-base-v2")
RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
DEFAULT_RETRIEVAL_K = int(os.environ.get("RAG_TOP_K", "10"))
DEFAULT_SIMILARITY_FLOOR = float(os.environ.get("RAG_SIMILARITY_FLOOR", "0.20"))
MAX_USER_CHARS = int(os.environ.get("CHAT_MAX_USER_CHARS", "8000"))
OPENAI_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
MODERATION_MODEL = os.environ.get("OPENAI_MODERATION_MODEL", "omni-moderation-latest")
ENABLE_CACHE = os.environ.get("RAG_ENABLE_CACHE", "true").lower() == "true"
CACHE_DIR = PROJECT_ROOT / ".cache"
LOG_PATH = PROJECT_ROOT / "logs" / "app.log"
FEEDBACK_PATH = PROJECT_ROOT / "logs" / "feedback.jsonl"


