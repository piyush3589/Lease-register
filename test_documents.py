"""Tests for PDF text extraction and its size limits."""

import pytest

import documents
from documents import (
    MAX_CHARS,
    MAX_PAGES,
    MAX_PDF_BYTES,
    DocumentTooLarge,
    text_from_pdf_bytes,
)

import make_sample_pdfs as gen


def sample_pdf_bytes(name="lease_residential_01"):
    with open(f"sample_leases/{name}.pdf", "rb") as fh:
        return fh.read()


def sample_pdf_text(name="lease_residential_01"):
    with open(f"sample_leases/{name}.txt", encoding="utf-8") as fh:
        return fh.read()


# --- failure modes ---------------------------------------------------------


def test_empty_bytes_raise():
    with pytest.raises(ValueError, match="empty"):
        text_from_pdf_bytes(b"")


def test_invalid_bytes_raise():
    with pytest.raises(ValueError, match="Could not read PDF"):
        text_from_pdf_bytes(b"not a pdf")


def test_pdf_with_no_text_layer_reports_the_ocr_limit():
    """A scanned lease is a capability gap, not a bug -- the message has to say so."""
    image_only = gen.build_pdf("\n\n")  # structurally valid, no drawable text
    with pytest.raises(ValueError, match="OCR"):
        text_from_pdf_bytes(image_only)


# --- happy path ------------------------------------------------------------


def test_generated_pdf_text_is_extracted():
    text = text_from_pdf_bytes(sample_pdf_bytes())
    assert "Priya Nair" in text
    assert "Arvind Shah" in text


def test_all_three_sample_pdfs_are_readable():
    for name in (
        "lease_residential_01",
        "lease_commercial_01",
        "lease_incomplete_01",
    ):
        text = text_from_pdf_bytes(sample_pdf_bytes(name))
        assert text.strip(), f"{name} produced no text"


def test_pdf_matches_its_source_txt():
    """If the generator drifts from the .txt fixtures, demos silently diverge."""
    extracted = text_from_pdf_bytes(sample_pdf_bytes())
    source = sample_pdf_text()
    assert "Priya Nair" in extracted and "Priya Nair" in source
    assert "Rs. 65,000" in extracted and "Rs. 65,000" in source


# --- limits ----------------------------------------------------------------


def test_oversize_pdf_bytes_raise():
    with pytest.raises(DocumentTooLarge, match="MB"):
        text_from_pdf_bytes(b"x" * (MAX_PDF_BYTES + 1))


def test_too_many_pages_raises(monkeypatch):
    monkeypatch.setattr(documents, "MAX_PAGES", 0)
    with pytest.raises(DocumentTooLarge, match="pages"):
        text_from_pdf_bytes(sample_pdf_bytes())


def test_too_much_extracted_text_raises(monkeypatch):
    monkeypatch.setattr(documents, "MAX_CHARS", 10)
    with pytest.raises(DocumentTooLarge, match="characters"):
        text_from_pdf_bytes(sample_pdf_bytes())


def test_document_too_large_is_a_value_error_subclass():
    """api.py catches ValueError for 422 and DocumentTooLarge for 413, so the
    subclass relationship is load-bearing."""
    assert issubclass(DocumentTooLarge, ValueError)


def test_limits_are_positive():
    assert MAX_PDF_BYTES > 0 and MAX_PAGES > 0 and MAX_CHARS > 0
