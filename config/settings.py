"""Application settings and environment configuration using Pydantic."""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Global configuration settings loaded from .env."""
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys for Groq and Google Gemini (3 keys each)
    GROQ_API_KEY1: str = Field(default="")
    GROQ_API_KEY2: str = Field(default="")
    GROQ_API_KEY3: str = Field(default="")
    GOOGLE_API_KEY1: str = Field(default="")
    GOOGLE_API_KEY2: str = Field(default="")
    GOOGLE_API_KEY3: str = Field(default="")

    # Model Defaults
    GROQ_MODEL1: str = "openai/gpt-oss-120b"
    GROQ_MODEL2: str = "openai/gpt-oss-20b"
    GEMINI_MODEL1: str = "gemini-3.6-flash"
    GEMINI_MODEL2: str = "gemini-3.5-flash-lite"

    # Execution Priority
    PRIMARY_PROVIDER: str = "groq"
    PROVIDER_CHAIN: str = "groq,gemini"

    # Directory Paths
    BASE_PATH: Path = BASE_DIR
    PDF_PATH: Path = BASE_DIR / "utils" / "pdf" / "medicare.pdf"
    DATA_PATH: Path = BASE_DIR / "data"
    CHROMA_PATH: Path = BASE_DIR / "data" / "cache" / "chroma_db"
    CONVERSATIONS_PATH: Path = BASE_DIR / "chat_conversations"
    LOGS_PATH: Path = BASE_DIR / "logs"

    # Dynamic Chunk Sizing Thresholds (characters)
    CHUNK_MIN_SIZE: int = 120
    CHUNK_MAX_SIZE: int = 650
    CHUNK_BASE_SIZE: int = 300
    ENTROPY_ALPHA: float = 0.35
    COHERENCE_BETA: float = 0.25

    # Retrieval & Session Configuration
    DEFAULT_TOP_K: int = 5
    MIN_CONFIDENCE_THRESHOLD: float = 0.40
    MAX_SESSION_TURNS: int = 20
    CONTEXT_WINDOW_PAIRS: int = 5

    def get_groq_keys(self) -> List[str]:
        """Return non-empty Groq API keys."""
        return [k.strip() for k in [self.GROQ_API_KEY1, self.GROQ_API_KEY2, self.GROQ_API_KEY3] if k.strip()]

    def get_gemini_keys(self) -> List[str]:
        """Return non-empty Gemini API keys."""
        return [k.strip() for k in [self.GOOGLE_API_KEY1, self.GOOGLE_API_KEY2, self.GOOGLE_API_KEY3] if k.strip()]

    def get_provider_list(self) -> List[str]:
        """Return ordered list of providers from PROVIDER_CHAIN."""
        return [p.strip().lower() for p in self.PROVIDER_CHAIN.split(",") if p.strip()]


settings = Settings()
