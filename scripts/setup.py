"""One-command project setup, PDF ingestion, and performance timing script.

Usage:
    uv run python scripts/setup.py
"""

import sys
import os
import shutil
import time
from pathlib import Path

# Enable UTF-8 console output for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import settings
from config.logger import logger


def setup_and_ingest():
    print("\n" + "=" * 80)
    print("[SETUP] MEDICARE RAG SYSTEM - INITIAL SETUP & INGESTION BENCHMARK")
    print("=" * 80)

    # 1. Check Python Version
    py_ver = sys.version_info
    print(f"Python Version: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    if py_ver < (3, 11):
        print("[ERROR] Python 3.11+ is required.")
        sys.exit(1)

    # 2. Check .env Configuration
    env_path = BASE_DIR / ".env"
    env_example_path = BASE_DIR / ".env.example"
    if not env_path.exists():
        if env_example_path.exists():
            shutil.copy(env_example_path, env_path)
            print("[WARN] Created .env from .env.example. Please populate your Groq/Gemini API keys!")
        else:
            print("[WARN] No .env file found. Please create one with your API keys.")
    else:
        print("[OK] Environment configuration (.env) detected.")

    # 3. Check PDF Document
    pdf_path = settings.PDF_PATH
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found at {pdf_path}")
        sys.exit(1)
    print(f"PDF Document: {pdf_path.name} ({pdf_path.stat().st_size / (1024*1024):.2f} MB)")

    # 4. Backup Existing ChromaDB if present
    chroma_dir = settings.CHROMA_PATH
    if chroma_dir.exists() and any(chroma_dir.iterdir()):
        backup_dir = chroma_dir.parent / "chroma_db_backup"
        print(f"Backing up existing ChromaDB to: {backup_dir.name}...")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(chroma_dir, backup_dir)
        print("[OK] ChromaDB backup created successfully.")

    # 5. Measure Ingestion & Dynamic Chunking Performance
    print("\nStarting fresh PDF extraction and algorithmic dynamic chunking...")
    t_start = time.perf_counter()

    from app.rag.ingestion.loader import pdf_loader
    from app.rag.chunking.dynamic_chunker import dynamic_chunker
    from app.rag.retrieval.vector_store import vector_store_manager

    # Step A: Load PDF
    t0 = time.perf_counter()
    documents = pdf_loader.load_pages()
    t1 = time.perf_counter()
    extract_time = t1 - t0
    print(f"  1. Extracted {len(documents)} pages from PDF in {extract_time:.2f}s")

    # Step B: Dynamic Chunking
    t2 = time.perf_counter()
    chunks = dynamic_chunker.split_documents(documents)
    t3 = time.perf_counter()
    chunk_time = t3 - t2
    print(f"  2. Algorithmic dynamic chunking produced {len(chunks)} chunks in {chunk_time:.2f}s")

    # Step C: ChromaDB Embedding & Storage
    t4 = time.perf_counter()
    total_vectors = vector_store_manager.add_documents(chunks, batch_size=32)
    t5 = time.perf_counter()
    embed_time = t5 - t4
    print(f"  3. FastEmbed embeddings & Chroma persistence completed in {embed_time:.2f}s")

    t_total = time.perf_counter() - t_start

    # 6. Summary Report
    summary_report = f"""
================================================================================
INGESTION PERFORMANCE & INVENTORY BENCHMARK REPORT
================================================================================
  Source PDF Document     : {pdf_path.name} ({pdf_path.stat().st_size / (1024*1024):.2f} MB)
  Total Pages Extracted   : {len(documents)} pages (1-indexed)
  Total Dynamic Chunks    : {len(chunks)} algorithmic chunks
  Total Chroma Vectors    : {total_vectors} embeddings persisted
--------------------------------------------------------------------------------
EXECUTION TIMING BREAKDOWN:
  1. PDF Page Extraction  : {extract_time:.3f}s
  2. Dynamic Chunk Sizing : {chunk_time:.3f}s
  3. ChromaDB Embedding   : {embed_time:.3f}s (FastEmbed ONNX)
  ------------------------------------------------------------------------------
  TOTAL SETUP DURATION    : {t_total:.2f}s ({t_total/60:.2f} minutes)
================================================================================
"""
    print(summary_report)
    logger.info(f"Setup and ingestion benchmark completed:\n{summary_report.strip()}")

    print("[SUCCESS] SETUP & INGESTION COMPLETE!")
    print("You can now start the FastAPI server with:")
    print("   uv run uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload")
    print("Or launch the interactive CLI chat with:")
    print("   uv run python main.py\n")



if __name__ == "__main__":
    setup_and_ingest()
