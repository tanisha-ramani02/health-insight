"""Unit tests for Chroma local vector storage and hybrid retrieval."""

import pytest
from langchain_core.documents import Document
from app.rag.retrieval.vector_store import VectorStoreManager
from app.rag.retrieval.hybrid_search import HybridSearchEngine



def test_vector_store_indexing_and_search(tmp_path):
    """Verify local Chroma storage and cosine similarity retrieval."""
    store = VectorStoreManager(persist_dir=tmp_path, collection_name="test_medicare")
    assert store.count() == 0

    docs = [
        Document(page_content="Medicare Open Enrollment begins on October 15 and ends on December 7.", metadata={"source_page": 15, "chunk_index": 1}),
        Document(page_content="Medicare Part D prescription drug coverage has a $2,000 cap in 2025.", metadata={"source_page": 82, "chunk_index": 2}),
    ]

    store.add_documents(docs)
    assert store.count() == 2

    # Search for enrollment
    results = store.similarity_search("enrollment deadlines", top_k=1)
    assert len(results) == 1
    top_doc, score = results[0]
    assert top_doc.metadata["source_page"] == 15
    assert score > 0.4

    # Test Hybrid Search
    engine = HybridSearchEngine(vector_mgr=store)
    hybrid_results = engine.search("What is the Part D drug cap in 2025?", top_k=1)
    assert len(hybrid_results) == 1
    h_doc, h_score = hybrid_results[0]
    assert h_doc.metadata["source_page"] == 82
