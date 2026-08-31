"""Interactive Command-Line Chat Interface for Medicare Policy Insight RAG.

Usage:
    uv run python main.py
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import settings
from config.logger import logger
from app.rag.ingestion.loader import pdf_loader
from app.rag.chunking.dynamic_chunker import dynamic_chunker
from app.rag.retrieval.hybrid_search import hybrid_engine
from app.rag.retrieval.vector_store import vector_store_manager
from utils.llm.router import llm_router
from utils.llm.scorer import confidence_scorer
from utils.llm.preprocessor import query_preprocessor
from app.session.manager import session_manager


def main():
    print("\n" + "=" * 75)
    print("🏥 Medicare Policy Intelligent RAG Assistant (CMS Handbook 2025)")
    print("=" * 75)
    print("Ask any question about Medicare Part A, B, C, D, deductibles, or 2025 rules.")
    print("Commands:")
    print("  - 'new'  : Start a fresh conversation session")
    print("  - 'exit' / 'quit' / 'bye' : End the session\n")

    # Ensure vector store is initialized
    if vector_store_manager.count() == 0:
        print("⏳ Initializing vector index from pdf/medicare.pdf (one-time setup)...")
        docs = pdf_loader.load_pages()
        chunks = dynamic_chunker.split_documents(docs)
        vector_store_manager.add_documents(chunks)
        print(f"✅ Indexed {len(chunks)} dynamic chunks into ChromaDB.\n")

    session_id, _ = session_manager.get_or_create_session()
    print(f"💬 Active Session ID: {session_id[:8]}... (Max {settings.MAX_SESSION_TURNS} turns per session)\n")

    while True:
        try:
            query = input("\n👤 You: ").strip()
            if not query:
                print("⚠️  Query cannot be empty. Please enter a valid question.")
                continue

            if query.lower() in ["exit", "quit", "bye", "q"]:
                print("\n👋 Thank you for using the Medicare Policy Assistant. Goodbye!\n")
                break

            if query.lower() == "new":
                session_id, _ = session_manager.get_or_create_session(None)
                print(f"\n🔄 Started fresh session: {session_id[:8]}...\n")
                continue

            t_start = datetime.now()
            print("🔍 Searching Medicare Handbook & Generating Answer...")

            # 1. Retrieve session history
            _, history = session_manager.get_or_create_session(session_id)
            current_turns = len(history) // 2
            if current_turns >= settings.MAX_SESSION_TURNS:
                print(f"\n⚠️  Session turn limit ({settings.MAX_SESSION_TURNS}) reached. Type 'new' to start a new chat.")
                continue

            # 2. Preprocess query (intent routing & pronoun resolution)
            prep_data = query_preprocessor.process(query, history)

            if prep_data.get("is_greeting") and prep_data.get("greeting_response"):
                t_end = datetime.now()
                ans = prep_data["greeting_response"]
                sid, turns, remaining, completed = session_manager.record_turn(
                    session_id=session_id,
                    query=query,
                    answer=ans,
                    start_time=t_start,
                    end_time=t_end,
                    source_page=None,
                    confidence_score=1.0,
                    chunk_size=0
                )
                latency_ms = round((t_end - t_start).total_seconds() * 1000, 1)

                print("\n🤖 Assistant:")
                print(f"   {ans}\n")
                print(f"   📄 Source Page: None (Conversational)")
                print(f"   📊 Confidence Score: 1.00 | 📏 Chunk Size: 0 chars")
                print(f"   ⚡ Latency: {latency_ms}ms | 💬 Turn: {turns}/{settings.MAX_SESSION_TURNS}")
                print("-" * 75)
                continue

            # 3. Hybrid search
            search_query = prep_data.get("search_query") or query
            history_str = session_manager.format_history_for_prompt(history)
            matched_chunks = hybrid_engine.search(search_query, top_k=settings.DEFAULT_TOP_K)

            if not matched_chunks:
                ans = "I could not find information regarding this topic in the official Medicare handbook."
                source_page = None
                conf = 0.0
                chunk_size = 0
                provider = "system:guardrail"
            else:
                top_doc, top_score = matched_chunks[0]
                source_page = top_doc.metadata.get("source_page", 1)
                chunk_size = top_doc.metadata.get("chunk_size", len(top_doc.page_content))
                context_str = "\n\n".join(
                    [f"[Source Page {d.metadata.get('source_page')}]: {d.page_content}" for d, _ in matched_chunks]
                )

                llm_output = llm_router.generate(query=query, context=context_str, chat_history_section=history_str)
                ans = llm_output.get("answer", "").strip()

                is_rejection = any(
                    p in ans.lower()
                    for p in ["could not find information", "not found in the official medicare handbook"]
                )

                if is_rejection:
                    source_page = None
                    conf = 0.0
                    chunk_size = 0
                else:
                    source_page = llm_output.get("source_page") or source_page
                    conf = confidence_scorer.compute(
                        retrieval_similarity=top_score,
                        hybrid_score=top_score,
                        answer=ans,
                        context=context_str
                    )

                provider = llm_output.get("_provider", "groq:primary")

            t_end = datetime.now()
            latency_ms = round((t_end - t_start).total_seconds() * 1000, 1)

            # Record turn
            sid, turns, remaining, completed = session_manager.record_turn(
                session_id=session_id,
                query=query,
                answer=ans,
                start_time=t_start,
                end_time=t_end,
                source_page=source_page,
                confidence_score=conf,
                chunk_size=chunk_size
            )

            # Display structured output
            print("\n🤖 Assistant:")
            print(f"   {ans}\n")
            print(f"   📄 Source Page: {f'Page {source_page}' if source_page else 'None (Out of Scope)'}")
            print(f"   📊 Confidence Score: {conf:.2f} | 📏 Dynamic Chunk Size: {chunk_size} chars")
            print(f"   ⚡ Latency: {latency_ms}ms | 🧠 Model: {provider} | 💬 Turn: {turns}/{settings.MAX_SESSION_TURNS}")
            print("-" * 75)

        except KeyboardInterrupt:
            print("\n\n👋 Session ended. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")


if __name__ == "__main__":
    main()
