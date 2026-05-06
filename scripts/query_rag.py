"""
Semantic search over the SQLite RAG index (same model as ingest).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from vector_store import SqliteRagIndex

PROJECT_ROOT = _SCRIPT_DIR.parent
DB_PATH = PROJECT_ROOT / "rag_data" / "index.sqlite"
EMBED_MODEL = "all-MiniLM-L6-v2"


def main() -> None:
    ap = argparse.ArgumentParser(description="Query RAG SQLite index")
    ap.add_argument("question", nargs="?", help="Question (or stdin)")
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("-k", type=int, default=5, help="Number of chunks to retrieve")
    ap.add_argument("--model", default=EMBED_MODEL)
    args = ap.parse_args()

    q = args.question
    if not q:
        q = sys.stdin.read().strip()
    if not q:
        print("Provide a question as an argument or via stdin.", file=sys.stderr)
        sys.exit(1)

    if not args.db.is_file():
        print(
            f"Index not found: {args.db}\nRun: python scripts/ingest_rag.py --reset --ocr all",
            file=sys.stderr,
        )
        sys.exit(1)

    index = SqliteRagIndex(args.db, model_name=args.model)
    if index.count() == 0:
        print(
            "Index is empty. Run: python scripts/ingest_rag.py --reset --ocr all",
            file=sys.stderr,
        )
        index.close()
        sys.exit(1)

    results = index.query(q, k=args.k)
    index.close()

    print("--- Retrieved context ---\n")
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        src = meta.get("source", "?")
        cidx = meta.get("chunk_index", "?")
        score = r["score"]
        doc = r["document"]
        cite = meta.get("citation") or meta.get("source", "")
        tag = meta.get("tag", "")
        p0 = meta.get("page_start", "")
        p1 = meta.get("page_end", "")
        ex = meta.get("extraction", "")
        line = f"[{i}] {src} | chunk {cidx} | score={score:.4f}"
        if tag:
            line += f" | tag={tag}"
        if p0 != "" and p1 != "":
            line += f" | pages {p0}-{p1}"
        if ex:
            line += f" | extraction={ex}"
        print(line)
        if cite and cite != src:
            print(f"  citation: {cite}")
        show = doc[:1500] + ("..." if len(doc) > 1500 else "")
        print(show)
        print()

    if results:
        best = results[0]["document"]
        snippet = best[:400].replace("\n", " ").strip()
        print("--- Suggested answer (from top chunk; add an LLM for a full chat) ---")
        print(snippet + ("..." if len(best) > 400 else ""))


if __name__ == "__main__":
    main()
