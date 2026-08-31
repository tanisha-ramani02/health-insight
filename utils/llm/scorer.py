"""Calibrated confidence scoring model combining retrieval similarity, normalized RRF, and lexical grounding."""

import re
from typing import List, Optional


class ConfidenceScorer:
    """Computes a calibrated confidence score for generated RAG answers."""

    def __init__(self, w_sim: float = 0.35, w_rrf: float = 0.35, w_ground: float = 0.30):
        self.w_sim = w_sim
        self.w_rrf = w_rrf
        self.w_ground = w_ground

    def _grounding_overlap(self, answer: str, context: str) -> float:
        """Measure what percentage of factual words in the answer exist in the context."""
        ans_words = set(re.findall(r"\b\w{4,}\b", answer.lower()))
        if not ans_words:
            return 0.5
        ctx_words = set(re.findall(r"\b\w{4,}\b", context.lower()))
        overlap = len(ans_words & ctx_words)
        return min(1.0, overlap / len(ans_words))

    def compute(
        self,
        retrieval_similarity: float,
        hybrid_score: float,
        answer: str,
        context: str
    ) -> float:
        """Calculate composite confidence score bounded between 0.0 and 1.0."""
        grounding = self._grounding_overlap(answer, context)

        # Normalize RRF score (max theoretical RRF for rank 1 in dense + rank 1 in BM25 is ~0.0328)
        norm_rrf = min(1.0, hybrid_score / 0.030) if hybrid_score < 0.2 else hybrid_score
        norm_sim = retrieval_similarity if retrieval_similarity > 0.2 else norm_rrf

        score = (self.w_sim * norm_sim) + (self.w_rrf * norm_rrf) + (self.w_ground * grounding)
        return round(max(0.0, min(1.0, score)), 2)


confidence_scorer = ConfidenceScorer()
