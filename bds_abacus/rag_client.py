"""RAG retrieval + OpenAI chat with guardrails, citations, and optional streaming."""
from __future__ import annotations

import sys
import time
import hashlib
import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Iterator, TypedDict
import diskcache

# Local vector store lives in scripts/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from vector_store import ChromaRagIndex  # noqa: E402

from bds_abacus import config
from bds_abacus import prompts
from bds_abacus import guardrails


@dataclass
class SourceInfo:
    index: int
    citation: str
    source: str
    page_start: int | str
    page_end: int | str
    score: float
    chunk_excerpt: str
    id: str = ""


@dataclass
class ChatResult:
    answer: str
    sources: list[SourceInfo]
    blocked: bool = False
    block_reason: str | None = None
    flags: list[str] = field(default_factory=list)
    tokens: int = 0
    time_taken: float = 0.0
    confidence: float = 0.0
    is_cached: bool = False


@dataclass
class StreamOutcome:
    """Result of `stream_chat`: either a token stream with sources, or a blocked `result`."""

    stream: Iterator[str] | None
    sources: list[SourceInfo] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    result: ChatResult | None = None  # if set, do not read stream
    tokens: int = 0
    time_taken: float = 0.0
    confidence: float = 0.0
    is_cached: bool = False


def _format_context_block(
    i: int,
    r: dict[str, Any],
) -> str:
    meta = r.get("metadata") or {}
    cite = meta.get("citation") or meta.get("source", "unknown")
    src = meta.get("source", "")
    p0, p1 = meta.get("page_start", ""), meta.get("page_end", "")
    head = f"[{i}] ({cite})"
    if src and src != cite:
        head += f" file={src}"
    if p0 != "" and p1 != "":
        head += f" pages {p0}-{p1}"
    body = (r.get("document") or "").strip()
    return f"{head}\n{body}"


def _rows_to_sources(results: list[dict[str, Any]]) -> list[SourceInfo]:
    out: list[SourceInfo] = []
    for i, r in enumerate(results, start=1):
        meta = r.get("metadata") or {}
        doc = r.get("document") or ""
        excerpt = doc[:500] + ("..." if len(doc) > 500 else "")
        out.append(
            SourceInfo(
                index=i,
                citation=str(meta.get("citation") or meta.get("source", "?")),
                source=str(meta.get("source", "?")),
                page_start=meta.get("page_start", ""),
                page_end=meta.get("page_end", ""),
                score=float(r.get("score", 0.0)),
                chunk_excerpt=excerpt,
                id=str(r.get("id", "")),
            )
        )
    return out


class RagChatService:
    """
    OpenAI + SQLite RAG. Call `close()` when done to release the DB connection, or use context manager.
    """

    def __init__(
        self,
        *,
        openai_client: Any,
        chroma_path: Path | None = None,
        embed_model: str | None = None,
        rerank_model: str | None = None,
        use_moderation: bool = True,
    ) -> None:
        self._client = openai_client
        self._chroma_path = chroma_path or config.CHROMA_PATH
        self._embed_model = embed_model or config.EMBED_MODEL
        self._rerank_model = rerank_model or config.RERANK_MODEL
        self._use_moderation = use_moderation
        self._index: ChromaRagIndex | None = None
        self._cache = diskcache.Cache(str(config.CACHE_DIR)) if config.ENABLE_CACHE else None
        self._global_context: str = ""
        self._load_global_context()
        self._setup_logging()

    def _setup_logging(self) -> None:
        config.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("RagChatService")
        if not self._logger.handlers:
            handler = logging.FileHandler(config.LOG_PATH, encoding="utf-8")
            fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(fmt)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def log_interaction(self, query: str, result: ChatResult) -> None:
        log_entry = {
            "query": query,
            "answer": result.answer,
            "tokens": result.tokens,
            "time": result.time_taken,
            "confidence": result.confidence,
            "cached": result.is_cached,
            "sources": [s.id for s in result.sources]
        }
        self._logger.info(f"Interaction: {json.dumps(log_entry)}")

    def record_feedback(self, query: str, answer: str, score: int, metadata: dict | None = None) -> None:
        """Score is usually 1 for positive, -1 for negative."""
        config.FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": time.time(),
            "query": query,
            "answer": answer,
            "score": score,
            "metadata": metadata or {}
        }
        with open(config.FEEDBACK_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _load_global_context(self) -> None:
        if config.CONTEXT_DOC_PATH.is_file():
            try:
                # Context document is in UTF-16LE
                self._global_context = config.CONTEXT_DOC_PATH.read_text(encoding="utf-16le")
            except Exception:
                try:
                    self._global_context = config.CONTEXT_DOC_PATH.read_text(encoding="utf-8")
                except Exception:
                    pass

    def _get_cache_key(self, prompt: str, params: dict) -> str:
        data = f"{prompt}:{params}"
        return hashlib.sha256(data.encode()).hexdigest()

    @property
    def index(self) -> ChromaRagIndex:
        if self._index is None:
            # Note: Chroma creates the directory if it doesn't exist, but we check if it has content
            # Actually, we'll just check if the directory exists.
            if not self._chroma_path.is_dir():
                 # For Chroma, a missing dir just means an empty DB is created, but for our app
                 # we might want to warn if no docs are indexed.
                 pass
            self._index = ChromaRagIndex(
                self._chroma_path, 
                model_name=self._embed_model,
                rerank_model_name=self._rerank_model
            )
        return self._index

    def close(self) -> None:
        if self._index is not None:
            self._index.close()
            self._index = None
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    def __enter__(self) -> RagChatService:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _preflight(self, user_text: str) -> ChatResult | None:
        g = guardrails.check_input(user_text, max_chars=config.MAX_USER_CHARS)
        if not g.ok:
            if g.reason == "message_blocked_safety":
                msg = "I cannot help with that request for safety reasons."
            else:
                msg = "I cannot process this message. " + (g.reason or "Please shorten or rephrase.")
            return ChatResult(
                answer=msg,
                sources=[],
                blocked=True,
                block_reason=g.reason,
                flags=g.flags,
            )
        if self._use_moderation:
            flagged, mflags = guardrails.moderate_text_sync(
                self._client, g.text, config.MODERATION_MODEL
            )
            if flagged:
                return ChatResult(
                    answer="I cannot help with that message because it was flagged by content safety checks.",
                    sources=[],
                    blocked=True,
                    block_reason="moderation",
                    flags=mflags,
                )
        return None

    def _retrieve_for_answer(
        self, t: str, k: int, similarity_floor: float
    ) -> tuple[list[dict[str, Any]] | None, str | None, list[str]]:
        """
        Returns (results_for_model, err_kind, display_flags).
        err_kind: 'missing_index' | 'no_retrieval' | 'low_confidence' | None
        """
        if not self._chroma_path.is_dir() or self.index.count() == 0:
            return None, "missing_index", ["index_missing"]
        raw = self.index.query(t, k=k)
        if not raw:
            return None, "no_retrieval", ["no_retrieval"]
        results = [r for r in raw if float(r.get("score", 0.0)) >= similarity_floor]
        if not results:
            return raw, "low_confidence", ["low_confidence", "all_chunks_below_threshold"]
        return results, None, []

    def chat(
        self,
        user_text: str,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        k: int | None = None,
        similarity_floor: float | None = None,
    ) -> ChatResult:
        model = model or config.OPENAI_MODEL
        k = k if k is not None else config.DEFAULT_RETRIEVAL_K
        similarity_floor = (
            similarity_floor
            if similarity_floor is not None
            else config.DEFAULT_SIMILARITY_FLOOR
        )

        t = guardrails.normalize_user_text(user_text)
        pre = self._preflight(t)
        if pre is not None:
            return pre

        raw_or_results, err_kind, rflag = self._retrieve_for_answer(
            t, k=k, similarity_floor=similarity_floor
        )
        if err_kind == "missing_index":
            return ChatResult(
                answer="The knowledge base index is missing. Ingest documents first, then try again.",
                sources=[],
                blocked=True,
                block_reason="missing_index",
                flags=rflag,
            )
        if err_kind == "no_retrieval":
            return ChatResult(
                answer="I could not find any relevant passages in the knowledge base for your question. "
                "Try rephrasing, or add documents to the index.",
                sources=[],
                flags=rflag,
            )
        if err_kind == "low_confidence":
            assert raw_or_results is not None
            raw = raw_or_results
            return ChatResult(
                answer="The most similar passages in the knowledge base are still weakly related to your question "
                f"(score below {similarity_floor:.2f}). I should not guess—try a more specific question or lower the "
                "similarity threshold in settings.",
                sources=_rows_to_sources(raw[: min(3, len(raw))]),
                flags=rflag,
            )
        assert raw_or_results is not None
        results = raw_or_results
        
        # Calculate confidence score
        confidence = sum(r.get("score", 0.0) for r in results) / len(results) if results else 0.0
        
        cache_key = self._get_cache_key(t, {"model": model, "temp": temperature, "k": k, "floor": similarity_floor})
        if self._cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            return ChatResult(
                answer=cached["answer"],
                sources=_rows_to_sources(results),
                flags=rflag,
                tokens=cached["tokens"],
                time_taken=cached["time"],
                confidence=confidence,
                is_cached=True
            )

        start_time = time.perf_counter()
        context_blocks = [_format_context_block(i, r) for i, r in enumerate(results, start=1)]
        # Inject global context
        if self._global_context:
            context_blocks.insert(0, f"[0] (GLOBAL CONTEXT)\n{self._global_context}")
            
        user_content = prompts.build_user_content(t, context_blocks)
        try:
            comp = self._client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": prompts.SYSTEM_GROUNDED},
                    {"role": "user", "content": user_content},
                ],
            )
        except Exception as e:
            return ChatResult(
                answer="The model request failed. Check your API key, network, and model name, then try again.",
                sources=_rows_to_sources(results),
                blocked=True,
                block_reason=f"api_error: {e!r}",
                flags=["openai_error"],
                confidence=confidence
            )

        end_time = time.perf_counter()
        text = (comp.choices[0].message.content or "").strip()
        tokens = comp.usage.total_tokens if comp.usage else 0
        
        if self._cache:
            self._cache[cache_key] = {"answer": text, "tokens": tokens, "time": end_time - start_time}

        og = guardrails.check_output_sanity(text)
        if not og.ok:
            return ChatResult(
                answer="The model returned an empty response. Please try again.",
                sources=_rows_to_sources(results),
                flags=og.flags,
                confidence=confidence
            )
        return ChatResult(
            answer=og.text, 
            sources=_rows_to_sources(results), 
            flags=rflag,
            tokens=tokens,
            time_taken=end_time - start_time,
            confidence=confidence
        )

    def stream_chat(
        self,
        user_text: str,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        k: int | None = None,
        similarity_floor: float | None = None,
    ) -> StreamOutcome:
        model = model or config.OPENAI_MODEL
        k = k if k is not None else config.DEFAULT_RETRIEVAL_K
        similarity_floor = (
            similarity_floor
            if similarity_floor is not None
            else config.DEFAULT_SIMILARITY_FLOOR
        )

        t = guardrails.normalize_user_text(user_text)
        pre = self._preflight(t)
        if pre is not None:
            return StreamOutcome(stream=None, result=pre)

        raw_or_results, err_kind, rflag = self._retrieve_for_answer(
            t, k=k, similarity_floor=similarity_floor
        )
        if err_kind == "missing_index":
            return StreamOutcome(
                stream=None,
                result=ChatResult(
                    answer="The knowledge base index is missing. Ingest documents first, then try again.",
                    sources=[],
                    blocked=True,
                    block_reason="missing_index",
                    flags=rflag,
                ),
            )
        if err_kind == "no_retrieval":
            return StreamOutcome(
                stream=None,
                result=ChatResult(
                    answer="I could not find any relevant passages in the knowledge base for your question. "
                    "Try rephrasing, or add documents to the index.",
                    sources=[],
                    flags=rflag,
                ),
            )
        if err_kind == "low_confidence":
            assert raw_or_results is not None
            raw = raw_or_results
            show = _rows_to_sources(raw[: min(3, len(raw))])
            return StreamOutcome(
                stream=None,
                result=ChatResult(
                    answer="The most similar passages are still weakly related (below your similarity floor). "
                    "Try a more specific question or lower the threshold.",
                    sources=show,
                    flags=rflag,
                ),
            )
        assert raw_or_results is not None
        results = raw_or_results
        confidence = sum(r.get("score", 0.0) for r in results) / len(results) if results else 0.0
        
        context_blocks = [_format_context_block(i, r) for i, r in enumerate(results, start=1)]
        if self._global_context:
            context_blocks.insert(0, f"[0] (GLOBAL CONTEXT)\n{self._global_context}")
            
        user_content = prompts.build_user_content(t, context_blocks)
        sources = _rows_to_sources(results)

        def _gen() -> Generator[str, None, None]:
            nonlocal start_time
            try:
                stream = self._client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    stream=True,
                    messages=[
                        {"role": "system", "content": prompts.SYSTEM_GROUNDED},
                        {"role": "user", "content": user_content},
                    ],
                )
                for ev in stream:
                    ch = ev.choices[0]
                    if ch.delta and ch.delta.content:
                        yield ch.delta.content
            except Exception as e:
                yield f"\n\n[Error: request failed: {e!r}]"

        start_time = time.perf_counter()
        # Streaming doesn't give token usage easily in the response object without extra steps, 
        # so we'll leave it as 0 or estimated for now.
        return StreamOutcome(
            stream=_gen(), 
            sources=sources, 
            flags=rflag, 
            result=None,
            confidence=confidence,
            time_taken=0.0  # Will be calculated at the end of stream in UI
        )
