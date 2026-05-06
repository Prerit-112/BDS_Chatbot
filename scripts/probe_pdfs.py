"""Probe PDFs for native text vs scan (pypdf; no PyMuPDF)."""
from pathlib import Path

from pypdf import PdfReader

DATA = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    for pdf in sorted(DATA.glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf))
            n_pages = len(reader.pages)
            parts: list[str] = []
            for p in reader.pages:
                t = p.extract_text() or ""
                parts.append(t)
            full = "\n".join(parts)
            stripped = full.strip()
            tlen = len(stripped)
            alnum = sum(1 for c in stripped if c.isalnum())
            sample = stripped[:300].replace("\n", " ") if stripped else "<empty or whitespace only>"
            verdict = "native_text" if tlen > 200 and alnum > 100 else "likely_scan_or_poor"
            print(f"--- {pdf.name}")
            print(f"  pages: {n_pages} | chars: {tlen} | alnum: {alnum} | {verdict}")
            print(f"  sample: {sample!r}")
        except Exception as e:
            print(f"ERR {pdf.name}: {e}")


if __name__ == "__main__":
    main()
