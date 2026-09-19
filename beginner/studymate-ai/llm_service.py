"""
llm_service.py

Handles all communication with the LLM provider through OpenRouter.

Responsibilities:
    - Initialize the API client from an environment variable.
    - Send prompts to a free OpenRouter model.
    - Handle API/network errors gracefully.
    - Return clean, plain-text results to the UI layer.

This file intentionally contains NO Streamlit / UI code.
"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

try:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        AuthenticationError,
        OpenAI,
        PermissionDeniedError,
        RateLimitError,
    )
except ImportError:  # pragma: no cover
    OpenAI = None
    APIConnectionError = None
    APITimeoutError = None
    AuthenticationError = None
    PermissionDeniedError = None
    RateLimitError = None


# Load variables from .env
load_dotenv()


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

ENV_VAR_NAME = "OPENROUTER_API_KEY"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# This router selects only currently available FREE models.
MODEL_NAME = "openrouter/free"

MAX_OUTPUT_TOKENS = 2000

REQUEST_TIMEOUT_SECONDS = 60


@dataclass
class LLMResult:
    """
    Represents the outcome of an LLM API call.

    success: whether the call completed successfully.
    text: model response text.
    error: user-friendly error message on failure.
    """

    success: bool
    text: str = ""
    error: str = ""


def _get_api_key() -> Optional[str]:
    """Read the OpenRouter API key from the environment."""

    api_key = os.getenv(ENV_VAR_NAME)

    if api_key is None or api_key.strip() == "":
        return None

    return api_key.strip()


def is_configured() -> bool:
    """Return True if an OpenRouter API key is available."""

    return _get_api_key() is not None


def _build_client():
    """
    Build and return an OpenRouter-compatible OpenAI client.

    Raises:
        RuntimeError: if the SDK is missing or the API key is missing.
    """

    if OpenAI is None:
        raise RuntimeError(
            "The 'openai' Python package is not installed. "
            "Please run: pip install -r requirements.txt"
        )

    api_key = _get_api_key()

    if api_key is None:
        raise RuntimeError(
            f"No API key found. Please set the {ENV_VAR_NAME} "
            "environment variable in your local .env file."
        )

    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def generate_response(prompt: str) -> LLMResult:
    """
    Send a prompt to the free OpenRouter model.

    This function never exposes raw API errors or credentials to the UI.
    """

    # ----------------------------------------------------------------------
    # Configuration validation
    # ----------------------------------------------------------------------

    try:
        client = _build_client()

    except RuntimeError as config_error:
        return LLMResult(
            success=False,
            error=str(config_error),
        )

    # ----------------------------------------------------------------------
    # API request
    # ----------------------------------------------------------------------

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

    except AuthenticationError:
        return LLMResult(
            success=False,
            error=(
                "The OpenRouter API rejected the configured API key. "
                "Please check that OPENROUTER_API_KEY is correct and active."
            ),
        )

    except PermissionDeniedError:
        return LLMResult(
            success=False,
            error=(
                "OpenRouter denied access for this API key. "
                "Please check your OpenRouter account and key."
            ),
        )

    except RateLimitError:
        return LLMResult(
            success=False,
            error=(
                "The free OpenRouter request limit has been reached. "
                "Please wait and try again later."
            ),
        )

    except APITimeoutError:
        return LLMResult(
            success=False,
            error=(
                "The AI service took too long to respond. "
                "Please try again."
            ),
        )

    except APIConnectionError:
        return LLMResult(
            success=False,
            error=(
                "Could not connect to OpenRouter. "
                "Please check your internet connection and try again."
            ),
        )

    except Exception:
        return LLMResult(
            success=False,
            error=(
                "Something went wrong while contacting the AI service. "
                "Please try again."
            ),
        )

    # ----------------------------------------------------------------------
    # Extract response text
    # ----------------------------------------------------------------------

    try:
        if not response.choices:
            return LLMResult(
                success=False,
                error="The AI service returned no response. Please try again.",
            )

        message = response.choices[0].message

        final_text = (message.content or "").strip()

    except Exception:
        return LLMResult(
            success=False,
            error=(
                "Received an unexpected response format from the AI service. "
                "Please try again."
            ),
        )

    # ----------------------------------------------------------------------
    # Empty response validation
    # ----------------------------------------------------------------------

    if not final_text:
        return LLMResult(
            success=False,
            error=(
                "The AI service returned an empty response. "
                "Please try again."
            ),
        )

    return LLMResult(
        success=True,
        text=final_text,
    )