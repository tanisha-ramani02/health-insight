"""Content-adaptive dynamic chunking algorithm using entropy and sentence boundaries."""

import re
from typing import List, Optional
from langchain_core.documents import Document
from config.settings import settings
from config.logger import logger
from .entropy import calculate_entropy, calculate_coherence


class AdaptiveDynamicChunker:
    """Algorithmically computes dynamic chunk sizes based on text characteristics."""

    def __init__(
        self,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        base_size: Optional[int] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None
    ):
        self.min_size = min_size or settings.CHUNK_MIN_SIZE
        self.max_size = max_size or settings.CHUNK_MAX_SIZE
        self.base_size = base_size or settings.CHUNK_BASE_SIZE
        self.alpha = alpha or settings.ENTROPY_ALPHA
        self.beta = beta or settings.COHERENCE_BETA

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences while respecting common healthcare abbreviations."""
        # Clean regex splitting on sentence boundaries
        pattern = r"(?<=[.!?])\s+(?=[A-Z0-9\"'•\-–])"
        raw_splits = re.split(pattern, text)
        sentences = [s.strip() for s in raw_splits if s.strip()]
        return sentences if sentences else [text.strip()]

    def calculate_target_size(self, entropy: float, coherence: float) -> int:
        """Compute dynamic chunk size: higher entropy yields smaller, focused chunks."""
        # Mean entropy reference is ~4.0 bits for English healthcare text
        entropy_factor = 1.0 - (self.alpha * (entropy - 4.0) / 2.0)
        coherence_factor = 1.0 + (self.beta * (coherence - 0.5) / 0.5)
        raw_size = self.base_size * entropy_factor * coherence_factor
        clamped_size = max(self.min_size, min(self.max_size, int(raw_size)))
        return clamped_size

    def split_document(self, doc: Document) -> List[Document]:
        """Split a single Document into variable dynamic-sized Document chunks."""
        text = doc.page_content
        if not text:
            return []

        sentences = self.split_sentences(text)
        chunks = []
        current_sentences = []
        current_length = 0

        for sentence in sentences:
            current_sentences.append(sentence)
            current_length += len(sentence)
            candidate_text = " ".join(current_sentences)

            entropy = calculate_entropy(candidate_text)
            coherence = calculate_coherence(current_sentences)
            target_size = self.calculate_target_size(entropy, coherence)

            # Trigger dynamic split when target size reached or at section break
            if current_length >= target_size or sentence.startswith("Section ") or sentence.endswith(":"):
                chunk_str = candidate_text.strip()
                chunks.append(Document(
                    page_content=chunk_str,
                    metadata={
                        **doc.metadata,
                        "chunk_size": len(chunk_str),
                        "entropy": entropy,
                        "coherence": coherence,
                        "chunk_index": len(chunks) + 1
                    }
                ))
                current_sentences = []
                current_length = 0

        # Flush any trailing sentences
        if current_sentences:
            chunk_str = " ".join(current_sentences).strip()
            chunks.append(Document(
                page_content=chunk_str,
                metadata={
                    **doc.metadata,
                    "chunk_size": len(chunk_str),
                    "entropy": calculate_entropy(chunk_str),
                    "coherence": calculate_coherence(current_sentences),
                    "chunk_index": len(chunks) + 1
                }
            ))

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a collection of documents dynamically."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.split_document(doc))
        logger.info(f"Dynamically generated {len(all_chunks)} chunks from {len(documents)} documents.")
        return all_chunks


dynamic_chunker = AdaptiveDynamicChunker()
