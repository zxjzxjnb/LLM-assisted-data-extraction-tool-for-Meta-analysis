"""Ingest sanity gate: a usable PDF parses; an unusable one fails loud."""

import pytest

from metaextract.ingest import IngestError, ingest_pdf

fitz = pytest.importorskip("fitz")


def _pdf(tmp_path, text: str | None) -> "object":
    """A one-page PDF with (or without) a text layer."""
    doc = fitz.open()
    page = doc.new_page()
    if text:
        page.insert_text((72, 72), text)
    path = tmp_path / "paper.pdf"
    doc.save(path)
    doc.close()
    return path


def test_text_pdf_ingests_with_addressable_words(tmp_path):
    doc = ingest_pdf(_pdf(tmp_path, "soil organic carbon 12.3 0.45"))
    assert doc.n_pages == 1
    assert any(b.kind == "paragraph" and b.words for b in doc.blocks)


def test_scanned_like_pdf_raises_with_ocr_hint(tmp_path):
    # A page with no text layer (the scanned-PDF signal) must fail loud, not
    # silently produce an empty doc that caches as "nothing to locate".
    with pytest.raises(IngestError, match="no extractable text"):
        ingest_pdf(_pdf(tmp_path, None))


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_pdf(tmp_path / "does_not_exist.pdf")
