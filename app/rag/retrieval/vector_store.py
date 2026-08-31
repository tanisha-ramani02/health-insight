"""Local persistent ChromaDB vector store manager with FastEmbed embeddings."""

from pathlib import Path
from typing import List, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from fastembed import TextEmbedding
from langchain_core.documents import Document
from config.settings import settings
from config.logger import logger


class VectorStoreManager:
    """Manages local Chroma vector collection and dense similarity search."""

    def __init__(self, persist_dir: Optional[Path] = None, collection_name: str = "medicare_docs"):
        self.persist_dir = persist_dir or settings.CHROMA_PATH
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.embed_model = TextEmbedding()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        """Return total chunk embeddings stored."""
        return self.collection.count()

    def add_documents(self, documents: List[Document], batch_size: int = 16) -> int:
        """Embed and persist Document chunks in Chroma in safe batches."""
        if not documents:
            return 0

        logger.info(f"Embedding and indexing {len(documents)} chunks into Chroma in batches of {batch_size}...")

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_texts = [doc.page_content for doc in batch_docs]
            # Embed only the current batch
            batch_embeds = [emb.tolist() for emb in self.embed_model.embed(batch_texts)]
            batch_ids = [
                f"chunk_{doc.metadata.get('source_page', 0)}_{doc.metadata.get('chunk_index', idx)}"
                for idx, doc in enumerate(batch_docs, start=i)
            ]
            batch_metas = [doc.metadata for doc in batch_docs]

            self.collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeds,
                documents=batch_texts,
                metadatas=batch_metas
            )

        total = self.collection.count()
        logger.info(f"Chroma indexing complete. Total vectors in store: {total}")
        return total

    def similarity_search(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """Perform dense cosine similarity search."""
        if not query.strip() or self.collection.count() == 0:
            return []

        query_emb = list(self.embed_model.embed([query]))[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        matched_docs = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for doc_text, meta, dist in zip(docs, metas, distances):
                # Cosine distance to similarity conversion: sim = 1.0 - dist
                similarity = max(0.0, min(1.0, 1.0 - dist))
                doc_obj = Document(page_content=doc_text, metadata=meta)
                matched_docs.append((doc_obj, similarity))

        return matched_docs


vector_store_manager = VectorStoreManager()
