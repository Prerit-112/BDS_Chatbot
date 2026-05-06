"""
Ingest PDFs from data/ into a local SQLite + embedding index.

Text extraction: pypdf + optional EasyOCR (default --ocr all merges both on every
page for maximum text). Each chunk stores tag, citation, page range, and extraction
labels (pypdf | ocr | pypdf+ocr) for provenance. Use --reset to delete all chunks
before rebuilding.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import sys
from pathlib import Path

# Allow running as `python scripts/ingest_rag.py`
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from pdf_ocr_extract import extract_pdf_hybrid
from vector_store import ChromaRagIndex

PROJECT_ROOT = _SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PATH = PROJECT_ROOT / "rag_data" / "chroma_db"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBED_MODEL = "all-mpnet-base-v2"
_TAG_ORDER = ("ocr", "pypdf", "pypdf+ocr")


def _extraction_label(methods: set[str]) -> str:
    o = {k: i for i, k in enumerate(_TAG_ORDER)}
    return " | ".join(sorted(methods, key=lambda x: (o.get(x, 50), x)))


import re

def _chunk_provenance(
    text: str,
    page_spans: list[tuple[int, int, int, str]],
    size: int,
    overlap: int,
) -> list[tuple[str, int, int, str]]:
    """
    Sentence-aware chunks; for each, page range and extraction label from overlapping pages.
    Returns list of (chunk_text, page_start, page_end, extraction_label).
    """
    if not text.strip():
        return []

    # 1. Split into sentences (simple regex)
    # Python's re module requires fixed-width look-behind.
    # We split into separate look-behinds for different lengths or just use a simpler check.
    abbreviations = ["Mr", "Mrs", "Ms", "Dr", "Sr", "Jr", "vs", "Prof", "etc", "eg", "ie"]
    lookbehind_parts = [f"(?<!\\b{ab})" for ab in abbreviations]
    sentence_endings = "".join(lookbehind_parts) + r"\.(?=\s|[A-Z])"
    raw_sentences = re.split(f"({sentence_endings})", text)
    
    sentences = []
    for i in range(0, len(raw_sentences)-1, 2):
        sentences.append(raw_sentences[i] + raw_sentences[i+1])
    if len(raw_sentences) % 2 != 0:
        sentences.append(raw_sentences[-1])
    
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 2. Reconstruct mapping of sentences to original character offsets
    # This is needed to get the page provenance
    cursor = 0
    sentence_map = [] # list of (text, start, end)
    for s in sentences:
        start = text.find(s, cursor)
        if start == -1: # fallback
            start = cursor
        end = start + len(s)
        sentence_map.append((s, start, end))
        cursor = end

    # 3. Group sentences into chunks
    out: list[tuple[str, int, int, str]] = []
    current_chunk_sentences = []
    current_chunk_len = 0
    
    i = 0
    while i < len(sentence_map):
        s_text, s_start, s_end = sentence_map[i]
        
        if current_chunk_len + len(s_text) > size and current_chunk_sentences:
            # Finalize current chunk
            chunk_text = " ".join([sm[0] for sm in current_chunk_sentences])
            c_start = current_chunk_sentences[0][1]
            c_end = current_chunk_sentences[-1][2]
            
            p_lo: int | None = None
            p_hi: int | None = None
            methods: set[str] = set()
            for ps_start, ps_end, pnum, m in page_spans:
                if c_end <= ps_start or c_start >= ps_end:
                    continue
                methods.add(m)
                p_lo = pnum if p_lo is None else min(p_lo, pnum)
                p_hi = pnum if p_hi is None else max(p_hi, pnum)
            
            if p_lo is None: p_lo, p_hi = 1, 1
            label = _extraction_label(methods) if methods else "unknown"
            out.append((chunk_text, int(p_lo), int(p_hi), label))
            
            # Start next chunk with overlap
            # Find how many sentences to keep for overlap
            overlap_len = 0
            overlap_sentences = []
            for j in range(len(current_chunk_sentences)-1, -1, -1):
                if overlap_len + len(current_chunk_sentences[j][0]) <= overlap:
                    overlap_sentences.insert(0, current_chunk_sentences[j])
                    overlap_len += len(current_chunk_sentences[j][0])
                else:
                    break
            
            current_chunk_sentences = overlap_sentences
            current_chunk_len = overlap_len
            
        current_chunk_sentences.append(sentence_map[i])
        current_chunk_len += len(s_text)
        i += 1

    # Add last chunk
    if current_chunk_sentences:
        chunk_text = " ".join([sm[0] for sm in current_chunk_sentences])
        c_start = current_chunk_sentences[0][1]
        c_end = current_chunk_sentences[-1][2]
        p_lo, p_hi, methods = None, None, set()
        for ps_start, ps_end, pnum, m in page_spans:
            if c_end <= ps_start or c_start >= ps_end: continue
            methods.add(m)
            p_lo = pnum if p_lo is None else min(p_lo, pnum)
            p_hi = pnum if p_hi is None else max(p_hi, pnum)
        if p_lo is None: p_lo, p_hi = 1, 1
        label = _extraction_label(methods) if methods else "unknown"
        out.append((chunk_text, int(p_lo), int(p_hi), label))

    return out


def _source_path_for_meta(pdf: Path) -> str:
    try:
        return str(pdf.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(pdf).replace("\\", "/")


def _match_pattern(path: Path, pattern: str) -> bool:
    if pattern in ("", "*", "*.*"):
        return True
    return fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(path.stem, pattern)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB RAG index")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--chroma-path", type=Path, default=CHROMA_PATH)
    ap.add_argument("--reset", action="store_true", help="Clear the index before ingesting")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    ap.add_argument("--model", default=EMBED_MODEL)
    ap.add_argument(
        "--include",
        type=str,
        default="*",
        help="Only ingest PDFs whose filename matches this glob (e.g. e3e* or *issue*).",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default="Stats",
        help="Tag stored on every chunk (e.g. for filtering in your app).",
    )
    ap.add_argument(
        "--ocr",
        choices=["off", "auto", "all"],
        default="all",
        help="off=pypdf only; auto=OCR sparse pages; all=pypdf+OCR on every page (max data, slow).",
    )
    ap.add_argument(
        "--ocr-threshold",
        type=int,
        default=100,
        help="With --ocr auto: OCR if page pypdf text (stripped) is shorter than this.",
    )
    ap.add_argument("--ocr-scale", type=float, default=2.0, help="pypdfium2 render scale (higher = clearer, slower).")
    ap.add_argument("--ocr-gpu", action="store_true", help="Use GPU for EasyOCR if available.")
    args = ap.parse_args()

    if args.ocr == "all":
        print(
            "Note: --ocr all runs EasyOCR on every page; expect a long run on large sets.\n"
            "       Use --ocr auto for a faster pass that OCRs only sparse pages.\n",
            file=sys.stderr,
        )

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    pdfs = [p for p in sorted(data_dir.glob("*.pdf")) if _match_pattern(p, args.include)]
    if not pdfs:
        print(f"No matching PDFs in {data_dir} (include={args.include!r})", file=sys.stderr)
        sys.exit(1)

    if args.reset and args.chroma_path.is_dir():
        import shutil
        import time
        try:
            shutil.rmtree(args.chroma_path)
            print(f"Deleted old index directory: {args.chroma_path}", file=sys.stderr)
        except PermissionError:
            new_name = f"chroma_db_{int(time.time())}"
            new_path = args.chroma_path.parent / new_name
            print(
                f"Warning: {args.chroma_path} is in use. Using new index directory: {new_path}",
                file=sys.stderr
            )
            args.chroma_path = new_path
            pass

    index = ChromaRagIndex(args.chroma_path, model_name=args.model)
    # If we wanted to reset but couldn't delete the directory, clear the collection instead
    # We only do this if we are still using the original locked directory.
    if args.reset and args.chroma_path.is_dir() and "chroma_db_" not in args.chroma_path.name:
        index.clear()
        print("Cleared documents in the existing collection.", file=sys.stderr)

    # Incremental check: find what's already indexed
    existing_sources = set()
    if not args.reset and index.count() > 0:
        res = index.collection.get(include=["metadatas"])
        if res["metadatas"]:
            for m in res["metadatas"]:
                if m and "source" in m:
                    existing_sources.add(m["source"])

    all_ids: list[str] = []
    all_docs: list[str] = []
    all_meta: list[dict] = []
    ocr_mode_arg = args.ocr

    n_sources = 0
    for pdf in pdfs:
        if pdf.name in existing_sources:
            print(f"Skipping {pdf.name} (already indexed).", file=sys.stderr)
            continue

        print(f"Extracting: {pdf.name} ...", file=sys.stderr, flush=True)
        try:
            text, n_pages, ostats, _page_m, page_spans = extract_pdf_hybrid(
                pdf,
                ocr_mode=ocr_mode_arg,
                ocr_threshold=args.ocr_threshold,
                ocr_scale=args.ocr_scale,
                ocr_gpu=args.ocr_gpu,
            )
        except Exception as e:
            print(f"Skip (read error) {pdf.name}: {e}", file=sys.stderr)
            continue
        if len(text.strip()) < 50:
            print(
                f"Skip (still almost empty after extract): {pdf.name}",
                file=sys.stderr,
            )
            continue
        print(
            f"  -> {len(text):,} chars, {n_pages} pages, stats={ostats}",
            file=sys.stderr,
        )
        base = pdf.stem
        rows = _chunk_provenance(
            text, page_spans, args.chunk_size, args.overlap
        )
        for idx, (ch, p0, p1, ex_label) in enumerate(rows):
            uid = hashlib.sha256(
                f"{args.tag}:{base}:{ocr_mode_arg}:{idx}:{ch[:64]}".encode()
            ).hexdigest()[:32]
            all_ids.append(uid)
            all_docs.append(ch)
            citation = f"{pdf.name} pp. {p0}-{p1}"
            all_meta.append(
                {
                    "tag": args.tag,
                    "source": pdf.name,
                    "source_path": _source_path_for_meta(pdf),
                    "source_stem": base,
                    "citation": citation,
                    "page_start": p0,
                    "page_end": p1,
                    "extraction": ex_label,
                    "ingest_ocr_mode": ocr_mode_arg,
                    "chunk_index": idx,
                    "n_chunks": len(rows),
                    "n_pages_doc": n_pages,
                    "embedding_model": args.model,
                }
            )

    if not all_ids:
        if existing_sources:
            print(f"All matching PDFs are already indexed in {args.chroma_path}.", file=sys.stderr)
            print(f"Total chunks in index: {index.count()}", file=sys.stderr)
        else:
            print("No chunks were extracted from the PDFs.", file=sys.stderr)
        index.close()
        return

    batch = 64
    for i in range(0, len(all_ids), batch):
        index.upsert_batch(
            all_ids[i : i + batch],
            all_docs[i : i + batch],
            all_meta[i : i + batch],
        )

    n_new_sources = len({m["source"] for m in all_meta})
    print(
        f"Indexed {n_new_sources} new PDFs -> {len(all_ids)} new chunks into {args.chroma_path}."
    )
    print(f"Total chunks in index: {index.count()}")
    index.close()


if __name__ == "__main__":
    main()
