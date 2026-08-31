"""FastAPI application initialization with async lifespan startup indexing."""

from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from config.settings import settings
from config.logger import logger
from app.rag.ingestion.loader import pdf_loader
from app.rag.chunking.dynamic_chunker import dynamic_chunker
from app.rag.retrieval.vector_store import vector_store_manager
from api.routes import router

STATIC_DIR = Path(__file__).resolve().parent.parent / "app" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to ensure vector store is indexed and warmed on startup."""
    logger.info("Initializing Medicare RAG Microservice...")
    current_vectors = vector_store_manager.count()

    if current_vectors == 0:
        logger.info("Vector store is empty. Starting ingestion and dynamic chunking of medicare.pdf...")
        documents = pdf_loader.load_pages()
        chunks = dynamic_chunker.split_documents(documents)
        vector_store_manager.add_documents(chunks)
        logger.info(f"Ingestion complete! Indexed {len(chunks)} chunks into Chroma.")
    else:
        logger.info(f"Vector store already contains {current_vectors} indexed chunks.")

    yield
    logger.info("Shutting down Medicare RAG Microservice.")


app = FastAPI(
    title="Medicare Policy RAG & Insight Microservice",
    description="Algorithmic Dynamic Chunking RAG API for CMS Medicare & You 2025 Handbook.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local testing and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(router)


# Serve Web Chat UI at Root
@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
async def serve_chat_ui():
    """Serve the interactive Medicare RAG Web Chat Interface."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Medicare RAG API is running. Visit <a href='/docs'>/docs</a>.</h1>")
