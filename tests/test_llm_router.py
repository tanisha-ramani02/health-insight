"""Unit tests for multi-provider LLM router and confidence scoring."""

import pytest
from utils.llm.router import llm_router
from utils.llm.scorer import confidence_scorer



def test_confidence_scorer():
    """Verify confidence score calibration formula."""
    sim = 0.85
    hyb = 0.80
    ans = "Medicare enrollment starts October 15 and ends December 7."
    ctx = "Medicare Open Enrollment begins on October 15 and ends on December 7 each year."

    score = confidence_scorer.compute(
        retrieval_similarity=sim,
        hybrid_score=hyb,
        answer=ans,
        context=ctx
    )
    assert 0.70 <= score <= 1.0, f"Expected high confidence score, got {score}"


def test_llm_router_generation():
    """Verify LLM router generation with Groq primary and structured JSON output."""
    context = "Medicare Open Enrollment begins October 15 and ends December 7 each year. Source: Page 15."
    query = "What are the Medicare enrollment dates?"

    res = llm_router.generate(query=query, context=context)
    assert "answer" in res, "Response missing 'answer' field"
    assert "source_page" in res or res.get("source_page") is not None
    assert len(res["answer"]) > 10
