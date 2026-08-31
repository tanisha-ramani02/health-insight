"""Mathematical metrics for text complexity, Shannon entropy, and coherence."""

import math
import re
from collections import Counter
from typing import List


def calculate_entropy(text: str) -> float:
    """Calculate Shannon Information Entropy H(X) in bits."""
    if not text or not text.strip():
        return 0.0
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    counts = Counter(words)
    total = len(words)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return round(entropy, 4)


def calculate_coherence(sentences: List[str]) -> float:
    """Calculate semantic coherence score based on lexical overlap across adjacent sentences."""
    if not sentences or len(sentences) <= 1:
        return 1.0

    scores = []
    for i in range(len(sentences) - 1):
        s1_words = set(re.findall(r"\b\w{3,}\b", sentences[i].lower()))
        s2_words = set(re.findall(r"\b\w{3,}\b", sentences[i + 1].lower()))
        if not s1_words or not s2_words:
            scores.append(0.5)
            continue
        intersection = len(s1_words & s2_words)
        union = len(s1_words | s2_words)
        jaccard = intersection / union if union > 0 else 0.5
        # Scale to 0.4 - 1.0 range
        scores.append(min(1.0, 0.4 + (jaccard * 1.5)))

    return round(sum(scores) / len(scores), 4)
