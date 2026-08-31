"""Evaluation script for Algorithmic Dynamic Chunking & Information Entropy metrics."""

import sys
from pathlib import Path

# Enable UTF-8 for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.rag.ingestion.loader import pdf_loader
from app.rag.chunking.dynamic_chunker import dynamic_chunker
from app.rag.chunking.entropy import calculate_entropy, calculate_coherence



def evaluate_dynamic_chunking():
    print("=" * 80)
    print("[EVAL] EVALUATING ALGORITHMIC DYNAMIC CHUNKING & ENTROPY METRICS")
    print("=" * 80)

    pages = pdf_loader.load_pages()
    test_pages = [2, 15, 32, 55, 83]

    print(f"\nEvaluating {len(test_pages)} sample pages across different handbook sections...\n")
    print(f"{'Page':<6} | {'Chars':<7} | {'Chunks':<7} | {'Min Size':<9} | {'Max Size':<9} | {'Avg Size':<9} | {'Avg Entropy':<12} | {'Avg Coherence':<12}")
    print("-" * 85)

    all_sizes = []
    all_entropies = []
    all_coherences = []

    for page_num in test_pages:
        target_docs = [p for p in pages if p.metadata.get("source_page") == page_num]
        if not target_docs:
            continue
        page_doc = target_docs[0]
        chunks = dynamic_chunker.split_document(page_doc)

        sizes = [c.metadata.get("chunk_size", len(c.page_content)) for c in chunks]
        entropies = [c.metadata.get("entropy", 0.0) for c in chunks]
        coherences = [c.metadata.get("coherence", 1.0) for c in chunks]

        all_sizes.extend(sizes)
        all_entropies.extend(entropies)
        all_coherences.extend(coherences)

        min_s = min(sizes) if sizes else 0
        max_s = max(sizes) if sizes else 0
        avg_s = sum(sizes) / len(sizes) if sizes else 0.0
        avg_e = sum(entropies) / len(entropies) if entropies else 0.0
        avg_c = sum(coherences) / len(coherences) if coherences else 0.0

        print(f"{page_num:<6} | {len(page_doc.page_content):<7} | {len(chunks):<7} | {min_s:<9} | {max_s:<9} | {avg_s:<9.1f} | {avg_e:<12.3f} | {avg_c:<12.3f}")

        # Assertions
        assert len(chunks) > 0, f"Page {page_num} produced zero chunks"
        assert min_s != max_s or len(chunks) == 1, f"Chunk sizes should be dynamic and variable"
        assert 2.0 <= avg_e <= 6.0, f"Entropy {avg_e} outside expected bounds"
        assert 0.2 <= avg_c <= 1.0, f"Coherence {avg_c} outside expected bounds"

    print("-" * 85)
    overall_avg_size = sum(all_sizes) / len(all_sizes) if all_sizes else 0
    overall_avg_entropy = sum(all_entropies) / len(all_entropies) if all_entropies else 0
    overall_avg_coherence = sum(all_coherences) / len(all_coherences) if all_coherences else 0

    print(f"\n[SUMMARY METRICS across {len(all_sizes)} dynamic chunks]:")
    print(f"  - Chunk Size Range      : {min(all_sizes)} to {max(all_sizes)} characters (Adaptive)")
    print(f"  - Mean Dynamic Size     : {overall_avg_size:.1f} characters")
    print(f"  - Mean Shannon Entropy  : {overall_avg_entropy:.3f} bits")
    print(f"  - Mean Semantic Coherence: {overall_avg_coherence:.3f}")
    print("\n[RESULT] Dynamic Chunking Verification: PASSED [SUCCESS]\n")


if __name__ == "__main__":
    evaluate_dynamic_chunking()
