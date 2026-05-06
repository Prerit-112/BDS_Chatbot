"""
Hybrid PDF text: pypdf for normal pages; EasyOCR for sparse or image-based pages.
Uses pypdfium2 to rasterize pages (wheels, no system Poppler on Windows).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

_reader: Any = None


def _get_easyocr_reader(gpu: bool) -> Any:
    global _reader
    if _reader is None:
        import easyocr  # local import, heavy

        _reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    return _reader


def ocr_pil_image(image: "Image.Image", *, gpu: bool = False) -> str:
    import numpy as np

    r = _get_easyocr_reader(gpu=gpu)
    arr = np.array(image)
    if arr.size == 0:
        return ""
    res = r.readtext(arr, detail=1)
    if not res:
        return ""
    return "\n".join(str(x[1]) for x in res if len(x) > 1).strip()


def _render_page_pil(pdf_path: Path, page_index: int, scale: float) -> "Image.Image":
    import pypdfium2 as pdfium
    from PIL import Image

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        page = doc[page_index]
        bitmap = page.render(scale=scale)
        pil = bitmap.to_pil()
        if pil.mode not in ("RGB", "L"):
            pil = pil.convert("RGB")
        return pil
    finally:
        doc.close()


def ocr_page(
    pdf_path: Path, page_index: int, *, scale: float = 2.0, gpu: bool = False
) -> str:
    try:
        image = _render_page_pil(pdf_path, page_index, scale=scale)
    except Exception as e:
        print(
            f"OCR: render fail {pdf_path.name} p{page_index + 1}: {e}", file=sys.stderr
        )
        return ""
    return ocr_pil_image(image, gpu=gpu)


def _page_method_label(raw_pypdf: str, ocr_t: str) -> str:
    """How this page was represented: pypdf, ocr, or pypdf+ocr (merged body)."""
    t = raw_pypdf.strip() if raw_pypdf else ""
    o = ocr_t.strip() if ocr_t else ""
    if t and o:
        return "pypdf+ocr"
    if o:
        return "ocr"
    return "pypdf"


def extract_pdf_hybrid(
    pdf_path: Path,
    *,
    ocr_mode: str = "auto",
    ocr_threshold: int = 100,
    ocr_scale: float = 2.0,
    ocr_gpu: bool = False,
) -> tuple[str, int, dict[str, int], list[str], list[tuple[int, int, int, str]]]:
    """
    ocr_mode: 'off' | 'auto' | 'all'
    - off: pypdf only
    - auto: pypdf; OCR a page if stripped text is shorter than ocr_threshold
    - all: pypdf + OCR on every page (merged) for maximum recall from images
    Returns (full_text, n_pages, stats, per_page_extraction, page_spans) where
    per_page_extraction is pypdf | ocr | pypdf+ocr per page, and page_spans are
    (char_start, char_end, page_1based, method) for the joined full_text.
    """
    if ocr_mode == "off":
        text, n, parts, page_methods = pypdf_only(pdf_path)
        spans = page_spans_for_joined_text(parts, page_methods)
        return text, n, {"ocr_pages": 0, "pypdf_only_pages": n}, page_methods, spans

    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    n = len(reader.pages)
    pypdf_parts: list[str] = []
    for i in range(n):
        pypdf_parts.append(reader.pages[i].extract_text() or "")

    stats: dict[str, int] = {"ocr_pages": 0, "pypdf_pages": 0}
    out_pages: list[str] = []
    page_methods: list[str] = []

    for i, raw in enumerate(pypdf_parts):
        t = raw.strip()
        if ocr_mode == "all":
            print(f"  OCR {i + 1}/{n} ...", file=sys.stderr, flush=True)
            ocr_t = ocr_page(pdf_path, i, scale=ocr_scale, gpu=ocr_gpu)
            stats["ocr_pages"] += 1
            if t and ocr_t:
                out_pages.append(f"{t}\n\n{ocr_t}")
            elif ocr_t:
                out_pages.append(ocr_t)
            else:
                out_pages.append(raw)
            page_methods.append(_page_method_label(raw, ocr_t or ""))
            continue

        # auto
        if len(t) >= ocr_threshold:
            out_pages.append(raw)
            page_methods.append("pypdf")
            stats["pypdf_pages"] += 1
        else:
            print(
                f"  OCR {i + 1}/{n} (pypdf {len(t)} ch) ...", file=sys.stderr, flush=True
            )
            ocr_t = ocr_page(pdf_path, i, scale=ocr_scale, gpu=ocr_gpu)
            stats["ocr_pages"] += 1
            ocr_t = ocr_t.strip() if ocr_t else ""
            if ocr_t:
                out_pages.append(ocr_t)
                page_methods.append("ocr")
            else:
                out_pages.append(raw)
                page_methods.append("pypdf")

    full = "\n\n".join(out_pages)
    spans = page_spans_for_joined_text(out_pages, page_methods)
    return full, n, stats, page_methods, spans


def pypdf_only(pdf_path: Path) -> tuple[str, int, list[str], list[str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    parts = [reader.pages[i].extract_text() or "" for i in range(len(reader.pages))]
    n = len(parts)
    methods = ["pypdf"] * n
    return "\n\n".join(parts), n, parts, methods


def page_spans_for_joined_text(
    out_pages: list[str], page_methods: list[str]
) -> list[tuple[int, int, int, str]]:
    """
    Map character positions in "\\n\\n".join(out_pages) to 1-based page and extraction label.
    Each tuple: (char_start inclusive, char_end exclusive, page_1based, method).
    """
    sep = "\n\n"
    if len(out_pages) != len(page_methods):
        page_methods = list(page_methods) + ["pypdf"] * (len(out_pages) - len(page_methods))
    pos = 0
    spans: list[tuple[int, int, int, str]] = []
    for i, ptext in enumerate(out_pages):
        s, e = pos, pos + len(ptext)
        m = page_methods[i] if i < len(page_methods) else "pypdf"
        spans.append((s, e, i + 1, m))
        pos = e + (len(sep) if i < len(out_pages) - 1 else 0)
    return spans
