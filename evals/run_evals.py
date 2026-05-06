"""
Offline guardrail evals (no API key) + optional RAG + LLM checks.

  python evals/run_evals.py
  python evals/run_evals.py --llm        # needs OPENAI_API_KEY and rag_data/index.sqlite
  python evals/run_evals.py --judge      # LLM-as-judge on one smoke answer (--llm implied)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from bds_abacus import config
from bds_abacus import guardrails


def _load_golden() -> dict:
    p = Path(__file__).resolve().parent / "golden.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def run_input_guards() -> list[tuple[str, bool, str | None]]:
    results: list[tuple[str, bool, str | None]] = []
    data = _load_golden()
    for c in data["cases"]:
        cid = c["id"]
        u = guardrails.normalize_user_text(c["user"])
        exp = c.get("expect") or {}
        g = guardrails.check_input(u, max_chars=config.MAX_USER_CHARS)
        want_ok = exp.get("input_ok", True)
        ok = g.ok == want_ok
        if not ok:
            results.append((cid, False, f"expected input_ok={want_ok}, got ok={g.ok}, reason={g.reason}"))
            continue
        flags = exp.get("flags_contain") or []
        miss = [f for f in flags if f not in g.flags]
        if miss:
            results.append((cid, False, f"missing flags {miss}, got {g.flags}"))
        else:
            results.append((cid, True, None))

    # Oversize
    text = "a" * (config.MAX_USER_CHARS + 1)
    g = guardrails.check_input(text, max_chars=config.MAX_USER_CHARS)
    if g.ok or "length" not in g.flags:
        results.append(("oversized", False, f"expected length block, got ok={g.ok} flags={g.flags}"))
    else:
        results.append(("oversized", True, None))
    return results


def _has_bracket_cite(s: str) -> bool:
    return bool(re.search(r"\[\d+\]", s))


def run_llm_smoke(*, with_judge: bool) -> list[tuple[str, bool, str | None]]:
    from openai import OpenAI

    from bds_abacus.rag_client import RagChatService

    out: list[tuple[str, bool, str | None]] = []
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return [("llm_smoke", False, "OPENAI_API_KEY not set")]

    if not config.DB_PATH.is_file():
        return [("llm_smoke", False, f"index missing: {config.DB_PATH}")]

    q = "Summarize the main subject matter of the documents in 2-3 short sentences, and cite at least one source using [1] or [2]."
    client = OpenAI(api_key=key)
    with RagChatService(openai_client=client) as rag:
        r = rag.chat(q, temperature=0.1)

    if r.blocked:
        return [("llm_smoke", False, f"blocked: {r.block_reason} {r.flags}")]

    if not r.sources:
        return [("llm_smoke", False, "no sources in result")]

    if not _has_bracket_cite(r.answer):
        out.append(
            (
                "citation_format",
                False,
                "answer did not contain [n] style citation; tighten prompts or re-run",
            )
        )
    else:
        out.append(("citation_format", True, None))

    if with_judge:
        judge = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict grader. Reply with exactly one line: SCORE: <1-5> | NOTE: <short reason>. "
                    "Score 5=fully grounded in cited context; 1=hallucination or no grounding.",
                },
                {
                    "role": "user",
                    "content": f"Question: {q}\n\nAnswer:\n{r.answer}\n\nSources (titles only): { [s.citation for s in r.sources] }",
                },
            ],
        )
        jtxt = (judge.choices[0].message.content or "").strip()
        m = re.search(r"SCORE:\s*(\d+)", jtxt, re.I)
        score = int(m.group(1)) if m else 0
        if score < 3:
            out.append(("llm_judge", False, jtxt))
        else:
            out.append(("llm_judge", True, jtxt))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="RAG+chat smoke (needs key + index)")
    ap.add_argument(
        "--judge",
        action="store_true",
        help="LLM-as-judge on smoke answer (implies --llm)",
    )
    args = ap.parse_args()

    failed = 0
    for cid, ok, err in run_input_guards():
        if ok:
            print(f"PASS  {cid}")
        else:
            print(f"FAIL  {cid}: {err}")
            failed += 1

    if args.llm or args.judge:
        for cid, ok, err in run_llm_smoke(with_judge=args.judge):
            if ok:
                print(f"PASS  {cid}" + (f" — {err}" if err and cid == "llm_judge" else ""))
            else:
                print(f"FAIL  {cid}: {err}")
                failed += 1
            if err and cid == "llm_judge":
                print(f"     judge detail: {err}")
    if failed:
        print(f"\n{failed} check(s) failed.", file=sys.stderr)
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
