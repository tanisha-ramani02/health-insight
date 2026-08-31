"""Unit tests for dynamic adaptive chunking and entropy evaluation."""

import pytest
from langchain_core.documents import Document
from app.rag.chunking.dynamic_chunker import dynamic_chunker
from app.rag.chunking.entropy import calculate_entropy, calculate_coherence



def test_entropy_and_coherence():
    """Verify entropy and coherence calculations."""
    text = "Medicare Part A covers inpatient hospital stays, care in a skilled nursing facility, hospice care, and some home health care."
    entropy = calculate_entropy(text)
    assert entropy > 2.0, f"Expected entropy > 2.0, got {entropy}"

    sentences = [
        "Medicare Part A covers inpatient hospital stays.",
        "It also covers skilled nursing care and hospice care.",
        "Medicare Part B covers doctor visits and medical tests."
    ]
    coherence = calculate_coherence(sentences)
    assert 0.0 <= coherence <= 1.0


def test_dynamic_chunking_variability():
    """Verify that chunk sizes are dynamic (variable) and not pre-defined/static."""
    sample_text = (
        "Medicare is health insurance for people 65 or older. Certain people younger than 65 can also get Medicare. "
        "These include people with disabilities and people with End-Stage Renal Disease. "
        "Medicare Part A is hospital insurance. It helps cover inpatient care in hospitals and skilled nursing facilities. "
        "Medicare Part B is medical insurance. It covers doctors services and outpatient care. "
        "Section 1 details the enrollment rules. You must sign up during your Initial Enrollment Period. "
        "If you miss it, you may pay a late enrollment penalty for as long as you have Part B."
    )
    doc = Document(page_content=sample_text, metadata={"source_page": 1})
    chunks = dynamic_chunker.split_document(doc)

    assert len(chunks) >= 2, "Expected multiple dynamic chunks"
    sizes = [c.metadata["chunk_size"] for c in chunks]

    # Verify that sizes vary adaptively
    assert len(set(sizes)) > 1 or len(chunks) == 1, "Chunk sizes should not be rigidly identical"
    for s in sizes:
        assert 100 <= s <= 700, f"Chunk size {s} outside dynamic bounds"
