"""Session and chat history manager storing JSON files locally with a 20-turn sliding window."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from config.settings import settings
from config.logger import logger
from utils.llm.router import llm_router



class SessionManager:
    """Manages chat sessions, local JSON storage, transcripts, and 20-turn sliding-window lifecycle."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or settings.CONVERSATIONS_PATH
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_turns = settings.MAX_SESSION_TURNS
        self.context_window_pairs = settings.CONTEXT_WINDOW_PAIRS

    def _get_session_path(self, session_id: str) -> Path:
        """Return the JSON file path for a session ID."""
        for file in self.storage_dir.glob(f"session_{session_id}_*.json"):
            return file
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.storage_dir / f"session_{session_id}_{ts}.json"

    def get_or_create_session(self, session_id: Optional[str] = None) -> Tuple[str, List[dict]]:
        """Retrieve existing session history or create a new session."""
        if not session_id or not session_id.strip():
            session_id = str(uuid.uuid4())
            return session_id, []

        path = self._get_session_path(session_id)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return session_id, data.get("messages", [])
            except Exception as e:
                logger.error(f"Failed to read session {session_id}: {e}")
                return session_id, []
        return session_id, []

    def get_session_details(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full details and messages of a session."""
        path = self._get_session_path(session_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                turn_count = len(data.get("messages", [])) // 2
                data["turns_remaining"] = max(0, self.max_turns - turn_count)
                data["is_last_turn"] = (turn_count == self.max_turns - 1)
                data["is_completed"] = (turn_count >= self.max_turns)
                return data
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return None

    def list_all_sessions(self) -> List[Dict[str, Any]]:
        """Scan chat_conversations directory and return all sessions sorted newest first."""
        sessions = []
        for file in self.storage_dir.glob("session_*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    messages = data.get("messages", [])
                    turn_count = len(messages) // 2

                    # Extract preview title from first user query
                    first_query = "New Session"
                    for m in messages:
                        if m.get("role") == "user":
                            first_query = m.get("content", "")[:35] + ("..." if len(m.get("content", "")) > 35 else "")
                            break

                    sessions.append({
                        "session_id": data.get("session_id", file.stem.split("_")[1]),
                        "title": first_query,
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),
                        "turn_count": turn_count,
                        "turns_remaining": max(0, self.max_turns - turn_count),
                        "is_completed": (turn_count >= self.max_turns),
                        "total_messages": len(messages)
                    })
            except Exception as e:
                logger.warning(f"Error parsing session file {file.name}: {e}")

        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions

    def record_turn(
        self,
        session_id: str,
        query: str,
        answer: str,
        start_time: datetime,
        end_time: datetime,
        source_page: Optional[int],
        confidence_score: float,
        chunk_size: int
    ) -> Tuple[str, int, int, bool]:
        """Record a turn. Returns (session_id, turn_count, turns_remaining, is_completed)."""
        session_id, history = self.get_or_create_session(session_id)
        current_turns = len(history) // 2

        if current_turns >= self.max_turns:
            raise ValueError(f"Session {session_id} has reached its maximum limit of {self.max_turns} turns.")

        latency_ms = round((end_time - start_time).total_seconds() * 1000, 2)

        history.append({
            "role": "user",
            "content": query,
            "timestamp": start_time.isoformat()
        })
        history.append({
            "role": "assistant",
            "content": answer,
            "timestamp": end_time.isoformat(),
            "source_page": source_page,
            "confidence_score": confidence_score,
            "chunk_size": chunk_size,
            "latency_ms": latency_ms
        })

        turn_count = len(history) // 2
        turns_remaining = max(0, self.max_turns - turn_count)
        is_completed = (turn_count >= self.max_turns)

        payload = {
            "session_id": session_id,
            "created_at": history[0]["timestamp"] if history else start_time.isoformat(),
            "updated_at": end_time.isoformat(),
            "turn_count": turn_count,
            "turns_remaining": turns_remaining,
            "is_completed": is_completed,
            "messages": history
        }

        path = self._get_session_path(session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return session_id, turn_count, turns_remaining, is_completed

    def _group_into_pairs(self, messages: List[dict]) -> List[Tuple[dict, dict]]:
        """Group flat message history into (user, assistant) pairs."""
        pairs = []
        i = 0
        while i < len(messages):
            if messages[i].get("role") == "user":
                user_msg = messages[i]
                asst_msg = messages[i + 1] if (i + 1 < len(messages) and messages[i + 1].get("role") == "assistant") else None
                if asst_msg:
                    pairs.append((user_msg, asst_msg))
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        return pairs

    def format_history_for_prompt(self, messages: List[dict]) -> str:
        """Build sliding-window context: summarize older turns + include last N pairs verbatim."""
        if not messages:
            return ""

        pairs = self._group_into_pairs(messages)
        if not pairs:
            return ""

        # Case 1: Pairs within window limit -> Send all verbatim
        if len(pairs) <= self.context_window_pairs:
            lines = ["Prior Conversation History:"]
            for u, a in pairs:
                lines.append(f"User: {u.get('content', '')}")
                lines.append(f"Assistant: {a.get('content', '')}")
            return "\n".join(lines) + "\n"

        # Case 2: Exceeds window -> Summarize older pairs + Send last N pairs verbatim
        old_pairs = pairs[:-self.context_window_pairs]
        recent_pairs = pairs[-self.context_window_pairs:]

        old_transcript = []
        for u, a in old_pairs:
            old_transcript.append(f"User: {u.get('content', '')}")
            old_transcript.append(f"Assistant: {a.get('content', '')}")
        old_text = "\n".join(old_transcript)

        try:
            summary = llm_router.generate_raw(
                prompt=f"--- EARLIER CONVERSATION ---\n{old_text}\n\nSUMMARY:",
                system_prompt="You are a Medicare chat history summarizer. Provide a concise 2-3 bullet summary of the key Medicare topics and answers discussed."
            )
            if not summary:
                summary = "Earlier queries covered general Medicare topics."
        except Exception as e:
            logger.warning(f"Failed to auto-summarize old turns: {e}")
            summary = "Earlier conversation covered Medicare policy topics."

        lines = [f"Summary of Earlier Conversation:\n{summary.strip()}\n", "Recent Conversation Turns:"]
        for u, a in recent_pairs:
            lines.append(f"User: {u.get('content', '')}")
            lines.append(f"Assistant: {a.get('content', '')}")

        return "\n".join(lines) + "\n"


session_manager = SessionManager()
