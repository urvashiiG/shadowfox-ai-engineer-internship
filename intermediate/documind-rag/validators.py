"""
validators.py

Input validation for uploaded files and user queries.

Every function returns a (is_valid, error_message) tuple so calling code
(the Streamlit UI or the RAG pipeline) can decide how to surface the
problem to the user, without ever crashing the app.
"""

from typing import Optional, Tuple, List

SUPPORTED_EXTENSIONS = (".pdf", ".txt")
MAX_QUERY_LENGTH = 1000
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB per file, generous for text/PDF notes


def validate_uploaded_files(files: Optional[List]) -> Tuple[bool, Optional[str]]:
    """
    Validate a list of uploaded files (Streamlit UploadedFile objects, or
    any object exposing `.name` and `.size`/`getbuffer()`).

    Checks:
    - at least one file was provided
    - each file has a supported extension (.pdf / .txt)
    - each file is not empty
    - each file is not unreasonably large
    """
    if not files:
        return False, "Please upload at least one PDF or TXT file."

    for f in files:
        name = getattr(f, "name", "")
        if not name:
            return False, "One of the uploaded files has no filename."

        lower_name = name.lower()
        if not lower_name.endswith(SUPPORTED_EXTENSIONS):
            return False, (
                f"Unsupported file type for '{name}'. "
                f"Only PDF and TXT files are supported."
            )

        # Try to determine size without consuming the stream permanently.
        size = getattr(f, "size", None)
        if size is None:
            try:
                size = len(f.getbuffer())
            except Exception:
                size = None

        if size is not None:
            if size == 0:
                return False, f"'{name}' is empty. Please upload a non-empty file."
            if size > MAX_FILE_SIZE_BYTES:
                return False, f"'{name}' is too large (max {MAX_FILE_SIZE_BYTES // (1024*1024)} MB)."

    return True, None


def validate_extracted_text(text: str, filename: str) -> Tuple[bool, Optional[str]]:
    """Validate that extracted text from a document is non-empty and useful."""
    if text is None or not text.strip():
        return False, (
            f"No readable text could be extracted from '{filename}'. "
            f"It may be empty, scanned as images, or corrupted."
        )
    return True, None


def validate_query(query: Optional[str]) -> Tuple[bool, Optional[str]]:
    """Validate a user's natural-language question."""
    if query is None or not query.strip():
        return False, "Please enter a question before asking DocuMind."

    stripped = query.strip()

    if len(stripped) < 3:
        return False, "Your question is too short. Please provide more detail."

    if len(stripped) > MAX_QUERY_LENGTH:
        return False, (
            f"Your question is too long ({len(stripped)} characters). "
            f"Please limit it to {MAX_QUERY_LENGTH} characters."
        )

    return True, None
