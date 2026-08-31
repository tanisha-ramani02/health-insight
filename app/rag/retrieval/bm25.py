"""Zero-dependency, ultra-fast BM25 implementation for exact & fuzzy keyword retrieval."""

import math
import re
from collections import Counter
from typing import List, Dict


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\b\w{2,}\b", text.lower())


class BM25Index:
    """In-memory BM25 index over full chunk corpus."""

    def __init__(self, corpus_texts: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_texts)
        corpus_tokens = [tokenize(t) for t in corpus_texts]
        self.doc_lengths = [len(doc) for doc in corpus_tokens]
        self.avgdl = (sum(self.doc_lengths) / max(1, self.corpus_size)) if self.corpus_size else 1.0

        self.doc_freqs: Dict[str, int] = Counter()
        self.term_freqs: List[Counter] = [Counter(doc) for doc in corpus_tokens]

        for doc in corpus_tokens:
            for term in set(doc):
                self.doc_freqs[term] += 1

    def get_scores(self, query: str) -> List[float]:
        """Compute BM25 score array across all documents."""
        query_tokens = tokenize(query)
        scores = [0.0] * self.corpus_size
        if not query_tokens or not self.corpus_size:
            return scores

        for term in query_tokens:
            if term not in self.doc_freqs:
                continue
            df = self.doc_freqs[term]
            idf = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))
            for i in range(self.corpus_size):
                tf = self.term_freqs[i].get(term, 0)
                if tf > 0:
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lengths[i] / self.avgdl))
                    scores[i] += idf * ((tf * (self.k1 + 1.0)) / denom)

        return scores
