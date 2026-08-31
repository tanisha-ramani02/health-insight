"""FastAPI REST routes for Medicare RAG queries, evaluation, history, sessions, and health."""

from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from config.settings import settings
from config.logger import logger
from .schemas import (
    QueryRequest,
    QueryResponse,
    ChunkEvalRequest,
    ChunkEvalResponse,
    HealthResponse,
    SessionHistoryResponse,
    SessionListResponse,
    SessionSummaryItem,
    NewSessionResponse
)
from app.rag.ingestion.loader import pdf_loader
from app.rag.chunking.dynamic_chunker import dynamic_chunker
from app.rag.retrieval.hybrid_search import hybrid_engine
from app.rag.retrieval.vector_store import vector_store_manager
from utils.llm.router import llm_router
from utils.llm.scorer import confidence_scorer
from utils.llm.preprocessor import query_preprocessor
from app.session.manager import session_manager

router = APIRouter(prefix="/api/v1", tags=["Medicare RAG Operations"])


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """Core RAG retrieval and structured QA endpoint with sliding-window session management."""
    start_time = datetime.now()
    user_query = req.query.strip()

    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be blank.")

    # 1. Retrieve session history and enforce max turns limit
    session_id, history = session_manager.get_or_create_session(req.session_id)
    current_turns = len(history) // 2

    if current_turns >= settings.MAX_SESSION_TURNS:
        raise HTTPException(
            status_code=400,
            detail=f"Chat limit of {settings.MAX_SESSION_TURNS} turns reached for this session ({session_id}). Please start a new chat."
        )

    # 2. LLM Preprocessing (Intent handling for greetings/empathy, and contextualizing follow-ups)
    prep_data = query_preprocessor.process(user_query, history)

    if prep_data.get("is_greeting") and prep_data.get("greeting_response"):
        end_time = datetime.now()
        greeting_text = prep_data["greeting_response"]
        session_id, turn_count, turns_remaining, is_completed = session_manager.record_turn(
            session_id=session_id,
            query=user_query,
            answer=greeting_text,
            start_time=start_time,
            end_time=end_time,
            source_page=None,
            confidence_score=1.0,
            chunk_size=0
        )
        latency_ms = round((end_time - start_time).total_seconds() * 1000, 2)
        return QueryResponse(
            answer=greeting_text,
            source_page=None,
            confidence_score=1.0,
            chunk_size=0,
            session_id=session_id,
            turn_count=turn_count,
            turns_remaining=turns_remaining,
            is_last_turn=(turns_remaining == 1),
            is_completed=is_completed,
            metadata={"latency_ms": latency_ms, "provider_used": "llm:preprocessor_greeting"}
        )

    search_query = prep_data.get("search_query") or user_query
    history_str = session_manager.format_history_for_prompt(history)

    # 3. Hybrid Retrieval (Dense Vector + Full-Corpus BM25 with Index Suppression)
    matched_chunks = hybrid_engine.search(search_query, top_k=max(6, req.top_k))

    if not matched_chunks:
        end_time = datetime.now()
        fallback_ans = "I could not find information regarding this topic in the official Medicare handbook."
        session_id, turn_count, turns_remaining, is_completed = session_manager.record_turn(
            session_id=session_id,
            query=user_query,
            answer=fallback_ans,
            start_time=start_time,
            end_time=end_time,
            source_page=None,
            confidence_score=0.0,
            chunk_size=0
        )
        latency_ms = round((end_time - start_time).total_seconds() * 1000, 2)
        return QueryResponse(
            answer=fallback_ans,
            source_page=None,
            confidence_score=0.0,
            chunk_size=0,
            session_id=session_id,
            turn_count=turn_count,
            turns_remaining=turns_remaining,
            is_last_turn=(turns_remaining == 1),
            is_completed=is_completed,
            metadata={"latency_ms": latency_ms, "provider_used": "system:guardrail"}
        )

    # 4. Format context and generate grounded structured response
    top_doc, top_hybrid_score = matched_chunks[0]
    context_passages = "\n\n".join(
        [f"[Source Page {doc.metadata.get('source_page', 'Unknown')}]: {doc.page_content}" for doc, _ in matched_chunks]
    )

    llm_output = llm_router.generate(
        query=user_query,
        context=context_passages,
        chat_history_section=history_str,
        force_provider=req.provider
    )

    answer_text = llm_output.get("answer", "").strip()
    is_rejection = any(
        phrase in answer_text.lower()
        for phrase in ["could not find information", "not found in the official medicare handbook", "i cannot comply", "can’t comply"]
    )

    if is_rejection:
        source_page = None
        calibrated_conf = 0.0
        chunk_size = 0
    else:
        source_page = llm_output.get("source_page") or top_doc.metadata.get("source_page", 1)
        chunk_size = top_doc.metadata.get("chunk_size", len(top_doc.page_content))
        calibrated_conf = confidence_scorer.compute(
            retrieval_similarity=top_hybrid_score,
            hybrid_score=top_hybrid_score,
            answer=answer_text,
            context=context_passages
        )

    end_time = datetime.now()

    # 5. Record turn in session JSON file
    session_id, turn_count, turns_remaining, is_completed = session_manager.record_turn(
        session_id=session_id,
        query=user_query,
        answer=answer_text,
        start_time=start_time,
        end_time=end_time,
        source_page=source_page,
        confidence_score=calibrated_conf,
        chunk_size=chunk_size
    )

    latency_ms = round((end_time - start_time).total_seconds() * 1000, 2)
    logger.info(f"Query completed in {latency_ms}ms | Turn: {turn_count}/{settings.MAX_SESSION_TURNS} | Page: {source_page} | Provider: {llm_output.get('_provider')}")

    return QueryResponse(
        answer=answer_text,
        source_page=source_page,
        confidence_score=calibrated_conf,
        chunk_size=chunk_size,
        session_id=session_id,
        turn_count=turn_count,
        turns_remaining=turns_remaining,
        is_last_turn=(turns_remaining == 1),
        is_completed=is_completed,
        metadata={
            "latency_ms": latency_ms,
            "provider_used": llm_output.get("_provider"),
            "search_query_used": search_query,
            "candidate_pages": [d.metadata.get("source_page") for d, _ in matched_chunks]
        }
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions_endpoint():
    """List all saved chat sessions with metadata, titles, and turn status."""
    session_list = session_manager.list_all_sessions()
    items = [SessionSummaryItem(**s) for s in session_list]
    return SessionListResponse(total_sessions=len(items), sessions=items)


@router.get("/session/{session_id}", response_model=SessionHistoryResponse)
async def get_session_endpoint(session_id: str):
    """Retrieve full turn-by-turn history of a specific session."""
    data = session_manager.get_session_details(session_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    return SessionHistoryResponse(**data)


@router.post("/session/new", response_model=NewSessionResponse)
async def new_session_endpoint():
    """Start a fresh chat session and return the new session ID."""
    new_id, _ = session_manager.get_or_create_session(None)
    return NewSessionResponse(
        session_id=new_id,
        message=f"Fresh session created successfully. Max {settings.MAX_SESSION_TURNS} turns available."
    )


@router.post("/chunk-evaluation", response_model=ChunkEvalResponse)
async def chunk_evaluation_endpoint(req: ChunkEvalRequest):
    """Inspect algorithmic dynamic chunk sizing across a specific PDF page."""
    docs = pdf_loader.load_pages()
    matching = [d for d in docs if d.metadata.get("source_page") == req.page_number]

    if not matching:
        raise HTTPException(status_code=404, detail=f"Page {req.page_number} not found.")

    target_page = matching[0]
    chunks = dynamic_chunker.split_document(target_page)
    sizes = [c.metadata.get("chunk_size", len(c.page_content)) for c in chunks]
    entropies = [c.metadata.get("entropy", 0.0) for c in chunks]
    coherences = [c.metadata.get("coherence", 1.0) for c in chunks]

    return ChunkEvalResponse(
        page_number=req.page_number,
        total_characters=len(target_page.page_content),
        total_chunks=len(chunks),
        dynamic_chunk_sizes=sizes,
        avg_chunk_size=round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
        avg_entropy=round(sum(entropies) / len(entropies), 3) if entropies else 0.0,
        avg_coherence=round(sum(coherences) / len(coherences), 3) if coherences else 1.0
    )


@router.get("/health", response_model=HealthResponse)
async def health_endpoint():
    """Health check and vector database inspection."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        document=str(settings.PDF_PATH.name),
        total_pages=128,
        total_chunks=vector_store_manager.count(),
        primary_provider=settings.PRIMARY_PROVIDER,
        active_keys_count={
            "groq_keys": len(settings.get_groq_keys()),
            "gemini_keys": len(settings.get_gemini_keys())
        }
    )
