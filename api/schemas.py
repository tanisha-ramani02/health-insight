"""Pydantic data models and schemas for API requests, responses, and sessions."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming query request model."""
    query: str = Field(..., min_length=1, max_length=1000, description="User question")
    session_id: Optional[str] = Field(default=None, description="Optional session UUID")
    top_k: int = Field(default=5, ge=1, le=10, description="Candidate chunks count")
    provider: Optional[str] = Field(default=None, description="Force provider (groq/gemini)")


class QueryResponse(BaseModel):
    """Standardized response schema as required by Assignment.md."""
    answer: str = Field(..., description="Grounded, structured answer")
    source_page: Optional[int] = Field(None, description="1-indexed source PDF page")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence")
    chunk_size: int = Field(..., ge=0, description="Dynamic chunk size in characters")
    session_id: Optional[str] = Field(None, description="Active session ID")
    turn_count: Optional[int] = Field(None, description="Conversation turns in session (max 20)")
    turns_remaining: Optional[int] = Field(default=None, description="Turns left before 20-turn limit")
    is_last_turn: Optional[bool] = Field(default=False, description="True if only 1 turn remains")
    is_completed: Optional[bool] = Field(default=False, description="True if 20-turn maximum reached")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Diagnostics")


class ChatMessage(BaseModel):
    """Single chat turn record with timestamps."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    source_page: Optional[int] = None
    confidence_score: Optional[float] = None
    chunk_size: Optional[int] = None
    latency_ms: Optional[float] = None


class SessionHistoryResponse(BaseModel):
    """Session history payload."""
    session_id: str
    created_at: str
    updated_at: Optional[str] = None
    turn_count: int
    turns_remaining: int
    is_last_turn: bool
    is_completed: bool
    messages: List[ChatMessage]


class SessionSummaryItem(BaseModel):
    """Metadata summary of a single saved session for the sidebar list."""
    session_id: str
    title: str
    created_at: str
    updated_at: str
    turn_count: int
    turns_remaining: int
    is_completed: bool
    total_messages: int


class SessionListResponse(BaseModel):
    """List of all historical chat sessions."""
    total_sessions: int
    sessions: List[SessionSummaryItem]


class NewSessionResponse(BaseModel):
    """Response when starting a fresh session."""
    session_id: str
    message: str


class ChunkEvalRequest(BaseModel):
    """Request for inspecting dynamic chunk sizing on a specific page."""
    page_number: int = Field(..., ge=1, le=128, description="PDF page number (1-128)")


class ChunkEvalResponse(BaseModel):
    """Evaluation metrics for dynamic chunking on a page."""
    page_number: int
    total_characters: int
    total_chunks: int
    dynamic_chunk_sizes: List[int]
    avg_chunk_size: float
    avg_entropy: float
    avg_coherence: float


class HealthResponse(BaseModel):
    """System health and vector database inspection."""
    status: str
    version: str
    document: str
    total_pages: int
    total_chunks: int
    primary_provider: str
    active_keys_count: Dict[str, int]
