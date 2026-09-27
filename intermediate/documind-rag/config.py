"""
config.py

Centralized configuration for DocuMind AI.

All secrets are read from environment variables (via a local .env file,
loaded with python-dotenv). Non-secret configuration has safe defaults so
the application can start even before a .env file is created.

The API key is never printed or logged anywhere in this project.
"""

import os
from dotenv import load_dotenv

# Load variables from a local .env file if present. This does not raise
# an error if the file is missing, so the app can still start (with the
# LLM layer reporting a friendly "not configured" state instead).
load_dotenv()

# ---------------------------------------------------------------------------
# OpenRouter / LLM configuration
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL: str = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).strip()

# Free OpenRouter model router by default -> no inference cost.
MODEL_NAME: str = os.getenv("MODEL_NAME", "openrouter/free").strip()

# ---------------------------------------------------------------------------
# Embedding configuration (local, no API cost)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME: str = os.getenv(
    "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
).strip()

# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))          # characters
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))    # characters

# ---------------------------------------------------------------------------
# Retrieval configuration
# ---------------------------------------------------------------------------
TOP_K: int = int(os.getenv("TOP_K", "4"))
MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))


def is_llm_configured() -> bool:
    """Return True if an OpenRouter API key has been provided."""
    return bool(OPENROUTER_API_KEY)
