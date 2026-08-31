"""Unit tests for configuration and environment settings."""

import pytest
from config.settings import settings


def test_settings_loaded():
    """Verify that settings load without error and paths exist."""
    assert settings.PDF_PATH.exists(), f"PDF not found at {settings.PDF_PATH}"
    assert len(settings.get_groq_keys()) >= 1, "At least 1 Groq key must be present"
    assert len(settings.get_gemini_keys()) >= 1, "At least 1 Gemini key must be present"
    assert settings.PRIMARY_PROVIDER in ["groq", "gemini"]
    assert settings.MAX_SESSION_TURNS == 20

