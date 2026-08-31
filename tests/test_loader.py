"""Unit tests for PDF document loading and metadata extraction."""

import pytest
from app.rag.ingestion.loader import pdf_loader



def test_pdf_extraction():
    """Verify that PDF is extracted with all 128 pages and 1-indexed metadata."""
    docs = pdf_loader.load_pages()
    assert len(docs) == 128, f"Expected 128 pages, got {len(docs)}"

    first_page = docs[0]
    assert first_page.metadata["source_page"] == 1
    assert "Medicare" in first_page.page_content

    # Check page 15 (Enrollment section mentioned in Assignment.md)
    page_15 = docs[14]
    assert page_15.metadata["source_page"] == 15
    assert "Signing up for Medicare" in page_15.page_content or "Part A" in page_15.page_content
