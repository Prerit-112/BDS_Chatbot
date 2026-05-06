"""
Streamlit UI for the RAG chatbot. Features:
- Professional BSG branding (Blue/Gold/White theme)
- Response caching & performance tracking
- Confidence score display
- User feedback mechanism
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Project root on path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

import streamlit as st
from openai import OpenAI

from bds_abacus import config
from bds_abacus.rag_client import ChatResult, RagChatService, StreamOutcome

# --- CUSTOM UI STYLING ---
def apply_custom_styles():
    st.markdown("""
        <style>
        :root {
            --bsg-blue: #1E3A8A;
            --bsg-gold: #D4AF37;
            --bsg-light: #F3F4F6;
        }
        .main {
            background-color: var(--bsg-light);
        }
        .stChatMessage {
            border-radius: 15px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .stChatMessage[data-testid="stChatMessageAssistant"] {
            background-color: white !important;
            border-left: 5px solid var(--bsg-blue);
        }
        .stChatMessage[data-testid="stChatMessageUser"] {
            background-color: #E0F2FE !important;
            border-right: 5px solid var(--bsg-blue);
        }
        .confidence-chip {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .confidence-high { background-color: #DCFCE7; color: #166534; }
        .confidence-med { background-color: #FEF9C3; color: #854D0E; }
        .confidence-low { background-color: #FEE2E2; color: #991B1B; }
        
        .metadata-row {
            font-size: 0.75rem;
            color: #6B7280;
            display: flex;
            gap: 15px;
            margin-top: 8px;
            border-top: 1px solid #E5E7EB;
            padding-top: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

def _get_api_key() -> str:
    try:
        s = st.secrets
        if "OPENAI_API_KEY" in s and s["OPENAI_API_KEY"]:
            return str(s["OPENAI_API_KEY"]).strip()
    except (RuntimeError, FileNotFoundError, KeyError):
        pass
    v = os.environ.get("OPENAI_API_KEY", "").strip()
    if v:
        return v
    return ""

@st.cache_resource
def _openai_client(api_key: str) -> OpenAI:
    if not api_key:
        raise ValueError("Missing API key")
    return OpenAI(api_key=api_key)

def _render_sources(sources: list) -> None:
    if not sources:
        st.caption("No sources (retrieval or guardrail path).")
        return
    for s in sources:
        with st.expander(f"Source [{s.index}]: {s.citation} (score {s.score:.3f})"):
            st.text(f"file: {s.source} | pages: {s.page_start}-{s.page_end}")
            st.caption("Excerpt")
            st.markdown(s.chunk_excerpt or "—")

def _get_confidence_class(score: float) -> str:
    if score >= 0.7: return "confidence-high"
    if score >= 0.4: return "confidence-med"
    return "confidence-low"

def main() -> None:
    st.set_page_config(
        page_title="BDS Abacus AI",
        page_icon="🧬",
        layout="wide",
    )
    apply_custom_styles()
    
    st.title("🧬 BDS Abacus AI")
    st.caption("Advanced RAG Support for Bharat Soka Gakkai Data Management")

    with st.sidebar:
        st.image("https://www.bharatsokagakkai.org/wp-content/uploads/2021/04/bsg-logo-new.png", width=100)
        st.header("Control Center")
            
        with st.expander("⚙️ Model Settings", expanded=False):
            model = st.text_input("Chat model", value=config.OPENAI_MODEL)
            temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
            k = st.number_input("Retrieval K", min_value=1, max_value=20, value=config.DEFAULT_RETRIEVAL_K)
            floor = st.slider("Similarity floor", 0.0, 0.6, config.DEFAULT_SIMILARITY_FLOOR, 0.01)
            use_mod = st.toggle("Safety Guardrails", value=True)
            
        st.divider()
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.caption(f"**Index:** `{config.CHROMA_PATH.name}`")
        if config.CHROMA_PATH.is_dir():
            st.success("Knowledge Base: Ready")
        else:
            st.error("Knowledge Base: Missing")

    api_key = _get_api_key()
    if not api_key:
        st.error(
            "OpenAI API key not found. Set `OPENAI_API_KEY` in your `.env` file "
            "(or in `.streamlit/secrets.toml`) and restart the app."
        )
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    client = _openai_client(api_key)

    if "last_api_key" not in st.session_state:
        st.session_state.last_api_key = ""
        
    if "rag" not in st.session_state or st.session_state.last_api_key != api_key:
        if "rag" in st.session_state:
            st.session_state.rag.close()
        st.session_state.rag = RagChatService(
            openai_client=client,
            use_moderation=use_mod,
        )
        st.session_state.last_api_key = api_key
    else:
        st.session_state.rag._use_moderation = use_mod

    rag: RagChatService = st.session_state.rag

    # Display Chat History
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            if m["role"] == "assistant" and m.get("confidence") is not None:
                c_class = _get_confidence_class(m["confidence"])
                st.markdown(f'<span class="confidence-chip {c_class}">Confidence: {m["confidence"]:.1%}</span>', unsafe_allow_html=True)
            
            st.markdown(m["content"] or " ")
            
            if m.get("sources"):
                with st.expander("View Grounding Sources"):
                    _render_sources(m["sources"])
            
            if m["role"] == "assistant":
                # Metadata row
                st.markdown(f"""
                    <div class="metadata-row">
                        <span>⏱️ {m.get('time', 0):.2f}s</span>
                        <span>🏷️ {m.get('tokens', 0)} tokens</span>
                        <span>{'⚡ Cached' if m.get('cached') else '🌐 Live'}</span>
                    </div>
                """, unsafe_allow_html=True)
                
                cols = st.columns([1, 1, 10])
                with cols[0]:
                    if st.button("👍", key=f"up_{i}"):
                        rag.record_feedback(st.session_state.messages[i-1]["content"], m["content"], 1)
                        st.toast("Thank you for your feedback!")
                with cols[1]:
                    if st.button("👎", key=f"down_{i}"):
                        rag.record_feedback(st.session_state.messages[i-1]["content"], m["content"], -1)
                        st.toast("Feedback recorded. We'll improve.")

    # Input handling
    if prompt := st.chat_input("Ask about BDS procedures, data entry, or reports..."):
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )
        st.rerun()

    # Assistant response
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        prompt = st.session_state.messages[-1]["content"]
        with st.chat_message("assistant"):
            with st.spinner("Analyzing knowledge base..."):
                start_exec = time.perf_counter()
                out: StreamOutcome = rag.stream_chat(
                    prompt,
                    model=model,
                    temperature=temperature,
                    k=int(k),
                    similarity_floor=floor,
                )
                
                if out.result is not None:
                    r: ChatResult = out.result
                    st.markdown(r.answer)
                    ans_text = r.answer
                    ans_sources = r.sources
                    ans_conf = r.confidence
                    ans_tokens = r.tokens
                    ans_time = r.time_taken
                    ans_cached = r.is_cached
                    rag.log_interaction(prompt, r)
                elif out.stream is not None:
                    ans_text = st.write_stream(out.stream) or ""
                    ans_sources = out.sources
                    ans_conf = out.confidence
                    ans_tokens = out.tokens 
                    ans_time = time.perf_counter() - start_exec
                    ans_cached = out.is_cached
                    # Create a dummy ChatResult for logging stream outcome
                    rag.log_interaction(prompt, ChatResult(
                        answer=ans_text,
                        sources=ans_sources,
                        tokens=ans_tokens,
                        time_taken=ans_time,
                        confidence=ans_conf,
                        is_cached=ans_cached
                    ))
                else:
                    ans_text = "Unexpected error."
                    ans_sources = []
                    ans_conf = 0.0
                    ans_tokens = 0
                    ans_time = 0.0
                    ans_cached = False
                    st.error(ans_text)

            st.session_state.messages.append({
                "role": "assistant",
                "content": ans_text,
                "sources": ans_sources,
                "confidence": ans_conf,
                "tokens": ans_tokens,
                "time": ans_time,
                "cached": ans_cached
            })
            st.rerun()

if __name__ == "__main__":
    main()
