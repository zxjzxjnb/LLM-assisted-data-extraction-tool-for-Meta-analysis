"""Deterministic PDF ingest → :class:`SourceDoc` (Architecture v3.1 §8 M0a).

PyMuPDF reads the PDF into addressable blocks (paragraphs / figures) and the
word-level geometry beneath them. This is the geometric ground truth the cockpit
needs for click-to-jump-and-highlight; the LLM never produces coordinates.

Note on tables: ``find_tables()`` returns nothing on borderless journal tables
(no ruling lines) and emits structure-tree errors, so it is **not used**. Instead
each table *row* already arrives as its own text block, and every number keeps an
exact word bbox — which is what value-level verification actually needs. Logical
table grouping (which blocks form "Table 2") is a Tier-2 concern, layered on top
of this geometry later, not a deterministic-ingest one.
"""

from __future__ import annotations

import re
import hashlib
from pathlib import Path

from .sourcedoc import BBox, SourceBlock, SourceDoc, Word

_TEXT_BLOCK = 0  # PyMuPDF block-type code for text (1 = image)


class IngestError(ValueError):
    """A PDF that exists but cannot yield a usable :class:`SourceDoc`.

    Raised instead of returning a degenerate empty document, so a scanned /
    image-only PDF fails loudly at prepare time rather than silently flowing
    downstream as a cache with nothing to locate or highlight.
    """


def ingest_pdf(pdf_path: str | Path) -> SourceDoc:
    """Parse a PDF into a :class:`SourceDoc` with bbox-addressable blocks.

    Raises :class:`FileNotFoundError` if the path does not exist and
    :class:`IngestError` if the PDF has no extractable text layer (almost always
    a scanned / image-only PDF): the locator and the click-to-evidence highlight
    both need real text, so an empty parse is a hard error, not a quiet result.
    """
    import fitz  # PyMuPDF; imported lazily so the rest of the package needn't have it

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"{pdf_path}: PDF not found.")
    doc = fitz.open(pdf_path)

    blocks: list[SourceBlock] = []
    page_sizes: dict[int, tuple[float, float]] = {}

    try:
        for pno, page in enumerate(doc, start=1):
            page_sizes[pno] = (float(page.rect.width), float(page.rect.height))
            words = _page_words(page)
            _ingest_text_blocks(page, pno, blocks, words)
            _ingest_figures(page, pno, blocks)
    finally:
        doc.close()

    _check_usable(pdf_path, len(page_sizes), blocks)

    return SourceDoc(
        doc_id=pdf_path.stem,
        n_pages=len(page_sizes),
        page_sizes=page_sizes,
        blocks=blocks,
        source_sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
    )


def _check_usable(pdf_path: Path, n_pages: int, blocks: list[SourceBlock]) -> None:
    """Reject a parse that the cockpit cannot act on (v3.1: fail loud, not empty).

    Two structural failures, both indistinguishable from a usable doc once cached
    so they must be caught here: a file with no pages (empty / not a real PDF), and
    a file whose pages carry no addressable words (a scanned / image-only PDF —
    there is text *on the page* as pixels, but no text *layer* to locate against).
    """
    if n_pages == 0:
        raise IngestError(f"{pdf_path.name}: no pages — the file is empty or not a readable PDF.")
    text_words = sum(len(b.words) for b in blocks if b.kind == "paragraph")
    if text_words == 0:
        raise IngestError(
            f"{pdf_path.name}: no extractable text across {n_pages} page(s). This is almost "
            "always a scanned / image-only PDF. Run OCR to add a text layer first "
            "(e.g. `ocrmypdf in.pdf out.pdf`) and re-ingest — MetaExtract needs the text "
            "layer to locate variables and to highlight values in the source."
        )


def _page_words(page) -> list[Word]:
    """All words on the page as (text, bbox); the precise-highlight layer."""
    out: list[Word] = []
    for x0, y0, x1, y1, text, *_ in page.get_text("words"):
        text = text.strip()
        if text:
            out.append(Word(text=text, bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1)))
    return out


def _ingest_text_blocks(
    page, pno: int, out: list[SourceBlock], page_words: list[Word]
) -> None:
    for bi, b in enumerate(page.get_text("blocks")):
        x0, y0, x1, y1, text, _bno, btype = b
        if btype != _TEXT_BLOCK:
            continue
        text = _normalize_ws(text)
        if not text:
            continue
        bbox = BBox(x0=x0, y0=y0, x1=x1, y1=y1)
        contained = [w for w in page_words if bbox.contains_center_of(w.bbox)]
        out.append(
            SourceBlock(
                block_id=f"p{pno}.b{bi}",
                kind="paragraph",
                page=pno,
                bbox=bbox,
                text=text,
                words=contained,
            )
        )


def _ingest_figures(page, pno: int, out: list[SourceBlock]) -> None:
    """Image regions as figure blocks (jumpable; numbers come via §6 digitizer)."""
    seen: set[tuple] = set()
    for fi, info in enumerate(page.get_image_info(), start=1):
        rect = info.get("bbox")
        if not rect:
            continue
        key = tuple(round(v) for v in rect)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            SourceBlock(
                block_id=f"p{pno}.fig{fi}",
                kind="figure",
                page=pno,
                bbox=BBox.from_rect(rect),
            )
        )


def _normalize_ws(text: str) -> str:
    """Collapse the per-word newlines PyMuPDF emits into single spaces."""
    return re.sub(r"\s+", " ", (text or "")).strip()
