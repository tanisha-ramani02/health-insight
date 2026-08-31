"""True Full-Corpus Hybrid Retrieval combining Dense Vector Search and BM25 via Reciprocal Rank Fusion (RRF)."""

from typing import List, Tuple, Optional
from langchain_core.documents import Document
from config.settings import settings
from config.logger import logger
from .vector_store import VectorStoreManager, vector_store_manager
from .bm25 import BM25Index


class HybridSearchEngine:
    """True Hybrid retrieval combining full-corpus BM25 and dense vector search via RRF."""

    def __init__(self, vector_mgr: VectorStoreManager = vector_store_manager):
        self.vector_mgr = vector_mgr
        self._bm25_index: Optional[BM25Index] = None
        self._corpus_docs: List[str] = []
        self._corpus_metas: List[dict] = []
        self._init_bm25()

    def _init_bm25(self):
        """Build in-memory BM25 index over all indexed chunks in ChromaDB."""
        try:
            all_data = self.vector_mgr.collection.get(include=["documents", "metadatas"])
            self._corpus_docs = all_data.get("documents", [])
            self._corpus_metas = all_data.get("metadatas", [])
            if self._corpus_docs:
                self._bm25_index = BM25Index(self._corpus_docs)
                logger.info(f"Initialized BM25 index over {len(self._corpus_docs)} chunks.")
        except Exception as e:
            logger.warning(f"Could not build BM25 index: {e}")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve top chunks using Reciprocal Rank Fusion of Dense + BM25 scores."""
        query_clean = query.strip()
        if not query_clean:
            return []

        # Ensure BM25 is initialized
        if self._bm25_index is None:
            self._init_bm25()

        is_explicit_index_query = any(k in query_clean.lower() for k in ["index", "contents", "table of contents", "10050"])

        # 1. Dense Vector Search (top 30)
        dense_results = self.vector_mgr.similarity_search(query_clean, top_k=30)

        # 2. BM25 Search across full corpus (top 30)
        top_bm25_candidates = []
        if self._bm25_index and self._corpus_docs:
            bm25_scores = self._bm25_index.get_scores(query_clean)
            top_bm25_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True
            )[:30]
            for rank, idx in enumerate(top_bm25_indices, 1):
                if bm25_scores[idx] > 0.1:
                    top_bm25_candidates.append((idx, rank, bm25_scores[idx]))

        # 3. Reciprocal Rank Fusion (RRF)
        fused_scores = {}

        # Add Dense RRF Scores
        for rank, (doc, sim) in enumerate(dense_results, 1):
            key = f"chunk_{doc.metadata.get('source_page', 0)}_{doc.metadata.get('chunk_index', rank)}"
            rrf_score = 1.0 / (60 + rank)
            page_num = doc.metadata.get("source_page", 1)

            # Suppress Table of Contents & Index unless explicitly queried
            if not is_explicit_index_query and ((3 <= page_num <= 8) or (119 <= page_num <= 128)):
                rrf_score *= 0.2

            fused_scores[key] = {
                "doc": doc,
                "score": rrf_score,
                "dense_sim": sim
            }

        # Add BM25 RRF Scores
        for idx, rank, score in top_bm25_candidates:
            doc_text = self._corpus_docs[idx]
            meta = self._corpus_metas[idx]
            page_num = meta.get("source_page", 1)
            key = f"chunk_{meta.get('source_page', 0)}_{meta.get('chunk_index', idx)}"
            rrf_score = 1.0 / (60 + rank)

            if not is_explicit_index_query and ((3 <= page_num <= 8) or (119 <= page_num <= 128)):
                rrf_score *= 0.2

            if key in fused_scores:
                fused_scores[key]["score"] += rrf_score
            else:
                doc_obj = Document(page_content=doc_text, metadata=meta)
                fused_scores[key] = {
                    "doc": doc_obj,
                    "score": rrf_score,
                    "dense_sim": 0.5
                }

        # Sort descending by RRF score
        ranked_candidates = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        top_candidates = [(item["doc"], round(item["score"], 4)) for item in ranked_candidates[:top_k]]

        logger.debug(f"Hybrid RRF search for '{query_clean[:40]}...' returned {len(top_candidates)} chunks.")
        return top_candidates


hybrid_engine = HybridSearchEngine()
