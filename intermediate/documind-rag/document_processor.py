"""
document_processor.py

Stage 1 & 2 of the RAG pipeline: document ingestion (text extraction)
and chunking.

Supported formats: PDF (via pypdf) and TXT (UTF-8, with a safe fallback).

Each chunk produced by `chunk_text` carries metadata:
    {
        "source": filename,
        "chunk_id": integer,
        "page": page number when available (None for TXT / unknown)
    }
"""

import re
from typing import List, Dict, Any, Optional

from pypdf import PdfReader

from config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_pdf(file) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF file object, page by page.

    Returns a list of {"page": page_number, "text": page_text} dicts so
    page numbers can be preserved in downstream chunk metadata.

    Any per-page extraction failure is skipped rather than raised, so a
    single corrupted page does not abort the whole document.
    """
    pages: List[Dict[str, Any]] = []
    try:
        reader = PdfReader(file)
    except Exception as exc:
        raise ValueError(f"Could not read PDF file: {exc}") from exc

    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append({"page": i, "text": text})

    return pages


def extract_text_from_txt(file) -> List[Dict[str, Any]]:
    """
    Extract text from a TXT file object.

    Tries UTF-8 first, falls back to latin-1 to avoid hard failures on
    files with unusual encodings. TXT files have no page concept, so
    "page" is always None.
    """
    raw = file.read()
    if isinstance(raw, str):
        text = raw
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="replace")

    if not text.strip():
        return []
    return [{"page": None, "text": text}]


def extract_text_from_file(file, filename: str) -> List[Dict[str, Any]]:
    """
    Dispatch extraction based on file extension.

    `file` must be a file-like object positioned at the start of the
    stream (Streamlit's UploadedFile satisfies this).
    """
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif lower_name.endswith(".txt"):
        return extract_text_from_txt(file)
    else:
        raise ValueError(f"Unsupported file type: {filename}")


def _normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace/newlines into single spaces."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(
    pages: List[Dict[str, Any]],
    source: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Split extracted page text into overlapping, meaningfully-sized chunks.

    Each chunk dict has the shape:
        {
            "text": str,
            "source": filename,
            "chunk_id": int,
            "page": int or None,
        }

    - Whitespace is normalized before splitting.
    - Chunks shorter than a small minimum are dropped to avoid noise.
    - Overlap preserves context across chunk boundaries so relevant
      information near a split point is not lost to retrieval.
    """
    if not pages:
        return []

    if chunk_overlap >= chunk_size:
        # Guard against a misconfiguration that would cause infinite loops.
        chunk_overlap = max(0, chunk_size // 4)

    chunks: List[Dict[str, Any]] = []
    chunk_id = 0
    min_chunk_len = 20  # drop trivially small fragments

    for page_entry in pages:
        page_number: Optional[int] = page_entry.get("page")
        text = _normalize_whitespace(page_entry.get("text", ""))

        if not text:
            continue

        start = 0
        text_len = len(text)

        if text_len <= chunk_size:
            chunks.append(
                {
                    "text": text,
                    "source": source,
                    "chunk_id": chunk_id,
                    "page": page_number,
                }
            )
            chunk_id += 1
            continue

        while start < text_len:
            end = min(start + chunk_size, text_len)
            piece = text[start:end].strip()

            if len(piece) >= min_chunk_len:
                chunks.append(
                    {
                        "text": piece,
                        "source": source,
                        "chunk_id": chunk_id,
                        "page": page_number,
                    }
                )
                chunk_id += 1

            if end >= text_len:
                break

            start = end - chunk_overlap

    return chunks
