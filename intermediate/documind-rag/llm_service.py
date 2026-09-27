"""
llm_service.py

Generation layer: calls OpenRouter (an OpenAI-compatible API) using the
official OpenAI Python SDK purely as a compatible HTTP client.

Do NOT hardcode API keys here. The key is read from config.py, which in
turn reads it from the environment / local .env file.

All errors are caught and converted into structured result dicts rather
than being allowed to propagate as raw exceptions into the UI layer.
"""

from typing import Dict, Any

from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    APIError,
)

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL_NAME, is_llm_configured


def _get_client() -> OpenAI:
    return OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)


def generate_grounded_answer(prompt: str, model: str = None) -> Dict[str, Any]:
    """
    Send a grounded prompt to the OpenRouter LLM and return a structured
    result:

        {
            "success": bool,
            "answer": str or None,
            "error": str or None,
        }

    Handles (without raising):
      - missing API key
      - authentication errors
      - rate limiting
      - connection errors
      - generic API errors
      - empty / malformed responses
    """
    if not is_llm_configured():
        return {
            "success": False,
            "answer": None,
            "error": (
                "OpenRouter API key is not configured. Add OPENROUTER_API_KEY "
                "to your .env file to enable answer generation."
            ),
        }

    model_name = model or MODEL_NAME

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are DocuMind AI, a careful assistant that answers "
                        "questions using ONLY the document context it is given."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=800,
        )
    except AuthenticationError:
        return {
            "success": False,
            "answer": None,
            "error": "Authentication with OpenRouter failed. Please check your OPENROUTER_API_KEY.",
        }
    except RateLimitError:
        return {
            "success": False,
            "answer": None,
            "error": "OpenRouter rate limit reached. Please wait a moment and try again.",
        }
    except APIConnectionError:
        return {
            "success": False,
            "answer": None,
            "error": "Could not connect to OpenRouter. Please check your internet connection.",
        }
    except APIError as exc:
        return {
            "success": False,
            "answer": None,
            "error": f"OpenRouter API returned an error: {exc}",
        }
    except Exception as exc:  # final safety net, never crash the UI
        return {
            "success": False,
            "answer": None,
            "error": f"Unexpected error while generating an answer: {exc}",
        }

    try:
        choice = response.choices[0]
        answer = (choice.message.content or "").strip()
    except (IndexError, AttributeError):
        answer = ""

    if not answer:
        return {
            "success": False,
            "answer": None,
            "error": "The model returned an empty response. Please try again.",
        }

    return {"success": True, "answer": answer, "error": None}
