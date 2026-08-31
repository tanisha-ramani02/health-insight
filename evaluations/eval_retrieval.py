"""Evaluation script for Full-Corpus BM25 + Dense Vector Hybrid Retrieval with RRF."""

import sys
import time
from pathlib import Path

# Enable UTF-8 for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.rag.retrieval.hybrid_search import hybrid_engine
from app.rag.retrieval.vector_store import vector_store_manager



def evaluate_hybrid_retrieval():
    print("=" * 80)
    print("[EVAL] EVALUATING HYBRID DENSE + BM25 RETRIEVAL (RECIPROCAL RANK FUSION)")
    print("=" * 80)

    test_queries = [
        {
            "id": "R1",
            "query": "What is the new out-of-pocket maximum spending cap for Medicare Part D prescription drugs in 2025?",
            "expected_pages": [2, 81, 82, 83]
        },
        {
            "id": "R2",
            "query": "When is the Medicare Advantage Open Enrollment Period (MA OEP) each year?",
            "expected_pages": [71, 72, 80]
        },
        {
            "id": "R3",
            "query": "Does Medicare pay for elective cosmetic plastic surgery done solely to improve personal appearance?",
            "expected_pages": [55, 56]
        },
        {
            "id": "R4",
            "query": "What are the limitations on chiropractic coverage under Medicare Part B?",
            "expected_pages": [34, 35]
        },
        {
            "id": "R5",
            "query": "What physical therapy and occupational therapy benefits are covered by Part B?",
            "expected_pages": [47, 48, 49, 50]
        }
    ]

    total = len(test_queries)
    passed = 0

    print(f"\nTesting {total} policy queries for hybrid candidate retrieval accuracy...\n")
    print(f"{'ID':<4} | {'Query Snippet':<40} | {'Expected':<12} | {'Top Retrieved':<14} | {'Match':<7} | {'Latency':<9}")
    print("-" * 92)

    for item in test_queries:
        qid = item["id"]
        query = item["query"]
        expected = item["expected_pages"]

        t0 = time.perf_counter()
        results = hybrid_engine.search(query, top_k=6)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000

        retrieved_pages = [doc.metadata.get("source_page") for doc, _ in results]
        overlap = set(expected) & set(retrieved_pages)
        is_hit = len(overlap) > 0

        if is_hit:
            passed += 1

        snippet = query[:37] + ("..." if len(query) > 37 else "")
        status = "PASS [OK]" if is_hit else "FAIL [X]"
        top_str = str(retrieved_pages[:4])

        print(f"{qid:<4} | {snippet:<40} | {str(expected):<12} | {top_str:<14} | {status:<7} | {latency_ms:6.1f}ms")

    print("-" * 92)
    hit_rate = (passed / total) * 100
    print(f"\n[SUMMARY RETRIEVAL METRICS]:")
    print(f"  - Total Queries Tested : {total}")
    print(f"  - Successful Top-K Hits: {passed}/{total}")
    print(f"  - Retrieval Hit Rate   : {hit_rate:.1f}%")
    print(f"\n[RESULT] Hybrid Retrieval Verification: {'PASSED [OK]' if passed == total else 'NEEDS ATTENTION'}\n")
    assert hit_rate >= 80.0, f"Retrieval hit rate {hit_rate}% below acceptable threshold (80%)"


if __name__ == "__main__":
    evaluate_hybrid_retrieval()
