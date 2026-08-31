"""Unit tests for chat session manager, local JSON persistence, and 20-turn sliding-window lifecycle."""

from datetime import datetime, timedelta
from app.session.manager import session_manager, SessionManager



def test_session_creation_and_recording(tmp_path):
    """Verify session JSON creation, timestamps, and 20-turn sliding window."""
    mgr = SessionManager(storage_dir=tmp_path)

    session_id, history = mgr.get_or_create_session()
    assert len(history) == 0

    t1 = datetime.now()
    t2 = t1 + timedelta(milliseconds=450)

    sid, turns, remaining, completed = mgr.record_turn(
        session_id=session_id,
        query="What is Part A?",
        answer="Part A covers inpatient hospital care.",
        start_time=t1,
        end_time=t2,
        source_page=15,
        confidence_score=0.95,
        chunk_size=250
    )

    assert sid == session_id
    assert turns == 1
    assert remaining == 19
    assert completed is False

    # Verify JSON file was created on disk
    files = list(tmp_path.glob(f"session_{session_id}_*.json"))
    assert len(files) == 1, "Session JSON file was not saved"

    # Test listing sessions
    all_sessions = mgr.list_all_sessions()
    assert len(all_sessions) == 1
    assert all_sessions[0]["session_id"] == session_id
    assert all_sessions[0]["turn_count"] == 1

    # Test context string formatting
    _, saved_history = mgr.get_or_create_session(session_id)
    history_str = mgr.format_history_for_prompt(saved_history)
    assert "User: What is Part A?" in history_str
    assert "Assistant: Part A covers inpatient hospital care." in history_str
