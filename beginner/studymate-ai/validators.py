"""
validators.py

Reusable input validation functions for StudyMate AI.

Every user-facing feature must run its input through these validators
BEFORE any LLM API call is made. This keeps validation logic in one
place, makes it independently testable, and prevents wasted/failed
API calls caused by obviously bad input.

Each validation function returns a tuple: (is_valid: bool, error_message: str)
`error_message` is an empty string when `is_valid` is True.
"""

from typing import Tuple

# Sensible character limits for a beginner-level student tool.
# These keep prompts within a reasonable size and protect against
# accidental huge pastes (e.g. an entire textbook) that would blow
# past token limits or produce unusable results.
MIN_TEXT_LENGTH = 3
MAX_CONCEPT_LENGTH = 300
MAX_NOTES_LENGTH = 12000
MAX_QUIZ_TOPIC_LENGTH = 12000
MAX_ANSWER_LENGTH = 6000

# Allowed number of quiz questions. Kept small and fixed for a
# beginner-level project rather than an unbounded numeric field.
ALLOWED_QUIZ_COUNTS = (5, 10, 15)


def is_empty_or_whitespace(text: str) -> bool:
    """Return True if the given text is None, empty, or only whitespace."""
    if text is None:
        return True
    return len(text.strip()) == 0


def validate_text_input(text: str, field_name: str = "input", max_length: int = MAX_NOTES_LENGTH) -> Tuple[bool, str]:
    """
    Validate a generic block of user-supplied text.

    Checks performed:
      1. The text is not None/empty/whitespace-only.
      2. The text meets a minimum meaningful length.
      3. The text does not exceed the given maximum length.

    Args:
        text: The raw user input.
        field_name: Human-readable name used in error messages.
        max_length: Maximum allowed number of characters.

    Returns:
        (is_valid, error_message)
    """
    if is_empty_or_whitespace(text):
        return False, f"Please enter some {field_name} before continuing."

    stripped = text.strip()

    if len(stripped) < MIN_TEXT_LENGTH:
        return False, f"Your {field_name} is too short. Please provide a bit more detail."

    if len(stripped) > max_length:
        return False, (
            f"Your {field_name} is too long ({len(stripped)} characters). "
            f"Please shorten it to under {max_length} characters."
        )

    return True, ""


def validate_concept_input(text: str) -> Tuple[bool, str]:
    """Validate the 'Explain Concept' topic input."""
    return validate_text_input(text, field_name="concept or topic", max_length=MAX_CONCEPT_LENGTH)


def validate_notes_input(text: str) -> Tuple[bool, str]:
    """Validate the 'Summarize Notes' input."""
    return validate_text_input(text, field_name="notes", max_length=MAX_NOTES_LENGTH)


def validate_quiz_topic_input(text: str) -> Tuple[bool, str]:
    """Validate the 'Generate Quiz' topic/material input."""
    return validate_text_input(text, field_name="study material or topic", max_length=MAX_QUIZ_TOPIC_LENGTH)


def validate_answer_input(text: str) -> Tuple[bool, str]:
    """Validate the 'Improve Answer' input."""
    return validate_text_input(text, field_name="answer", max_length=MAX_ANSWER_LENGTH)


def validate_quiz_count(count) -> Tuple[bool, str]:
    """
    Validate the requested number of quiz questions.

    Accepts an int or a value convertible to int. Only values in
    ALLOWED_QUIZ_COUNTS are considered valid for this beginner-level
    project (5, 10, or 15 questions).

    Args:
        count: The requested quiz question count.

    Returns:
        (is_valid, error_message)
    """
    try:
        count_int = int(count)
    except (TypeError, ValueError):
        return False, "Please select a valid number of quiz questions."

    if count_int not in ALLOWED_QUIZ_COUNTS:
        allowed = ", ".join(str(c) for c in ALLOWED_QUIZ_COUNTS)
        return False, f"Please choose one of the supported quiz sizes: {allowed} questions."

    return True, ""
