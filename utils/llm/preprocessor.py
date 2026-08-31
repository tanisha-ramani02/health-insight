"""LLM-driven query preprocessor for conversational routing, capability guidance, and query rewriting.

Handles:
1. Greetings, pleasantries, empathy, and emotional statements.
2. Assistant capability and scope inquiries (e.g. 'what all can you help me with', 'so what can you do?').
3. Non-medical general chit-chat (e.g. 'about some life philosophy', 'tell me a joke').
4. Resolving pronouns (it, this, that) and rewriting follow-ups into standalone search queries.
"""

import json
import re
from typing import Dict, Any, List
from config.logger import logger
from .router import llm_router



PREPROCESSOR_PROMPT = """You are a smart preprocessor and conversational router for the official Medicare & You 2025 Handbook Assistant.
Given the previous chat history and the user's latest query, classify and handle:

1. GREETING / CAPABILITY / SCOPE / CHIT-CHAT:
   - If the user asks a greeting (e.g., 'hi', 'how are you', 'good morning', 'thanks'),
   - Or asks what you can do / capability / guidance (e.g., 'so what can you?', 'what all you can help me with', 'what do you cover?', 'i want guidance'),
   - Or makes an emotional statement (e.g., 'i feel tensed', 'i am overwhelmed'),
   - Or asks non-medical general chit-chat (e.g., 'about some life philosophy', 'tell me a joke'):
   -> Set is_greeting=true and provide a warm, empathetic, professional response explaining your role as a Medicare Assistant and what topics you can help with (Part A/B/C/D, 2025 $2000 drug cap, enrollment deadlines, deductibles).

2. MEDICARE QUESTION / CONVERSATIONAL FOLLOW-UP:
   - If the user is asking about Medicare policies or following up on previous turns (e.g., 'What is Part A?', 'more details about it in 4 points', 'summarize it in 4 lines', 'Does Medicare pay for cosmetic surgery?'):
   -> Set is_greeting=false, resolve any pronouns (it, this, that, etc.) using the chat history, fix any typos, and output a concise standalone search query for vector retrieval.

Respond in strict JSON:
{
  "is_greeting": false,
  "greeting_response": null,
  "search_query": "standalone search query"
}
"""


class QueryPreprocessor:
    """Fast LLM-based preprocessor for intent classification and contextualization."""

    def process(self, query: str, history: List[dict]) -> Dict[str, Any]:
        """Classify intent and rewrite query using prior context if needed."""
        clean_q = query.strip()
        if not clean_q:
            return {"is_greeting": False, "greeting_response": None, "search_query": clean_q}

        # Format recent history turns
        history_snippet = []
        for msg in history[-4:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")[:200]
            history_snippet.append(f"{role}: {content}")
        hist_text = "\n".join(history_snippet) if history_snippet else "None"

        user_content = f"--- CHAT HISTORY ---\n{hist_text}\n\n--- USER QUERY ---\n{clean_q}\n\nStrict JSON response:"

        try:
            raw_text = llm_router.generate_raw(prompt=user_content, system_prompt=PREPROCESSOR_PROMPT)
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                logger.info(f"Preprocessor for '{clean_q}': is_greeting={data.get('is_greeting')}, query='{data.get('search_query')}'")
                return data
        except Exception as e:
            logger.warning(f"Preprocessor LLM call failed: {e}. Using raw query.")

        return {"is_greeting": False, "greeting_response": None, "search_query": clean_q}


query_preprocessor = QueryPreprocessor()
