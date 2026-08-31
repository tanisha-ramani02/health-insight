"""Multi-provider LLM router supporting 3 Groq keys and 3 Gemini keys with key rotation and failover."""

import json
import re
from typing import Any, Dict, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings
from config.logger import logger
from utils.prompts.prompts import SYSTEM_RAG_PROMPT, RAG_USER_PROMPT, STRICT_JSON_SCHEMA_INSTRUCTION




class LLMRouter:
    """Manages LLM execution with 3-key round-robin rotation and multi-provider failover."""

    def __init__(self):
        self.groq_keys = settings.get_groq_keys()
        self.gemini_keys = settings.get_gemini_keys()
        self.groq_idx = 0
        self.gemini_idx = 0

    def _get_next_groq_key(self) -> Optional[str]:
        """Round-robin selection of Groq API key."""
        if not self.groq_keys:
            return None
        key = self.groq_keys[self.groq_idx % len(self.groq_keys)]
        self.groq_idx = (self.groq_idx + 1) % len(self.groq_keys)
        return key

    def _get_next_gemini_key(self) -> Optional[str]:
        """Round-robin selection of Gemini API key."""
        if not self.gemini_keys:
            return None
        key = self.gemini_keys[self.gemini_idx % len(self.gemini_keys)]
        self.gemini_idx = (self.gemini_idx + 1) % len(self.gemini_keys)
        return key

    def _extract_json(self, raw_text: str) -> Dict[str, Any]:
        """Extract and parse structured JSON from LLM output."""
        cleaned = raw_text.strip()
        # Remove markdown code fences if present
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback regex extraction
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
            logger.warning(f"Raw output could not be parsed as JSON: {raw_text[:100]}")
            return {"answer": raw_text, "source_page": None, "confidence_score": 0.5, "chunk_size": 0}

    def _call_groq(self, prompt: str, model_name: str) -> Dict[str, Any]:
        """Invoke Groq LLM with round-robin key rotation."""
        for attempt in range(len(self.groq_keys)):
            key = self._get_next_groq_key()
            try:
                logger.debug(f"Calling Groq model '{model_name}' (Key attempt {attempt + 1})...")
                llm = ChatGroq(api_key=key, model=model_name, temperature=0.0)
                messages = [
                    SystemMessage(content=SYSTEM_RAG_PROMPT + "\n" + STRICT_JSON_SCHEMA_INSTRUCTION),
                    HumanMessage(content=prompt)
                ]
                res = llm.invoke(messages)
                content = res.content if isinstance(res.content, str) else str(res.content)
                parsed = self._extract_json(content)
                parsed["_provider"] = f"groq:{model_name}"
                return parsed
            except Exception as e:
                logger.warning(f"Groq invocation failed on key {key[:8]}...: {e}")
        raise RuntimeError("All Groq API keys failed or were rate-limited.")

    def _call_gemini(self, prompt: str, model_name: str) -> Dict[str, Any]:
        """Invoke Google Gemini LLM with round-robin key rotation."""
        for attempt in range(len(self.gemini_keys)):
            key = self._get_next_gemini_key()
            try:
                logger.debug(f"Calling Gemini model '{model_name}' (Key attempt {attempt + 1})...")
                llm = ChatGoogleGenerativeAI(google_api_key=key, model=model_name)
                messages = [
                    SystemMessage(content=SYSTEM_RAG_PROMPT + "\n" + STRICT_JSON_SCHEMA_INSTRUCTION),
                    HumanMessage(content=prompt)
                ]
                res = llm.invoke(messages)
                # Handle list of parts if returned by Gemini
                if isinstance(res.content, list):
                    text_parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in res.content]
                    content = "".join(text_parts)
                else:
                    content = str(res.content)
                parsed = self._extract_json(content)
                parsed["_provider"] = f"gemini:{model_name}"
                return parsed
            except Exception as e:
                logger.warning(f"Gemini invocation failed on key {key[:8]}...: {e}")
        raise RuntimeError("All Gemini API keys failed or were rate-limited.")

    def generate(
        self,
        query: str,
        context: str,
        chat_history_section: str = "",
        chat_history_str: str = "",
        force_provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate structured response with automatic provider priority and fallback."""
        hist_str = chat_history_section or chat_history_str or ""
        prompt = RAG_USER_PROMPT.format(
            context=context,
            chat_history_section=hist_str,
            query=query
        )


        providers = [force_provider.lower()] if force_provider else settings.get_provider_list()

        for provider in providers:
            if provider == "groq":
                try:
                    return self._call_groq(prompt, settings.GROQ_MODEL1)
                except Exception as e:
                    logger.error(f"Groq primary failed: {e}. Trying secondary Groq model...")
                    try:
                        return self._call_groq(prompt, settings.GROQ_MODEL2)
                    except Exception as e2:
                        logger.error(f"Groq secondary failed: {e2}. Falling back to Gemini...")

            elif provider == "gemini":
                try:
                    return self._call_gemini(prompt, settings.GEMINI_MODEL1)
                except Exception as e:
                    logger.error(f"Gemini primary failed: {e}. Trying secondary Gemini model...")
                    try:
                        return self._call_gemini(prompt, settings.GEMINI_MODEL2)
                    except Exception as e2:
                        logger.error(f"Gemini secondary failed: {e2}.")

        raise RuntimeError("All LLM providers and fallbacks failed to generate a response.")

    def generate_raw(self, prompt: str, system_prompt: str = "") -> str:
        """Execute a fast raw text generation call across active LLM providers."""
        providers = settings.get_provider_list()
        for provider in providers:
            if provider == "groq":
                for _ in range(len(self.groq_keys)):
                    key = self._get_next_groq_key()
                    try:
                        llm = ChatGroq(api_key=key, model=settings.GROQ_MODEL1, temperature=0.0)
                        msgs = []
                        if system_prompt:
                            msgs.append(SystemMessage(content=system_prompt))
                        msgs.append(HumanMessage(content=prompt))
                        res = llm.invoke(msgs)
                        return res.content if isinstance(res.content, str) else str(res.content)
                    except Exception as e:
                        logger.warning(f"Groq raw call failed: {e}")
            elif provider == "gemini":
                for _ in range(len(self.gemini_keys)):
                    key = self._get_next_gemini_key()
                    try:
                        llm = ChatGoogleGenerativeAI(google_api_key=key, model=settings.GEMINI_MODEL1)
                        msgs = []
                        if system_prompt:
                            msgs.append(SystemMessage(content=system_prompt))
                        msgs.append(HumanMessage(content=prompt))
                        res = llm.invoke(msgs)
                        return res.content if isinstance(res.content, str) else str(res.content)
                    except Exception as e:
                        logger.warning(f"Gemini raw call failed: {e}")
        return ""


llm_router = LLMRouter()

