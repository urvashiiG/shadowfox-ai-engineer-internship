from pathlib import Path
from typing import BinaryIO
from pypdf import PdfReader
from app.processing.cleaner import clean_text

SUPPORTED = {".pdf", ".txt", ".md"}

class DocumentLoadError(ValueError):
    """A user-correctable document ingestion error."""

def load_document(file: BinaryIO, filename: str, max_bytes: int = 15_000_000) -> list[dict]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise DocumentLoadError("Unsupported file type. Upload PDF, TXT, or Markdown files.")
    data = file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise DocumentLoadError("File is too large. The maximum upload size is 15 MB.")
    if not data:
        raise DocumentLoadError("The uploaded file is empty.")
    pages: list[dict] = []
    try:
        if suffix == ".pdf":
            import io
            reader = PdfReader(io.BytesIO(data), strict=True)
            for number, page in enumerate(reader.pages, start=1):
                pages.append({"text": clean_text(page.extract_text() or ""), "page_number": number})
        else:
            text = data.decode("utf-8-sig")
            pages.append({"text": clean_text(text), "page_number": None})
    except Exception as exc:
        raise DocumentLoadError("Could not read this file. Check that it is a valid, readable document.") from exc
    pages = [page for page in pages if page["text"]]
    if not pages:
        raise DocumentLoadError("No extractable text was found in this document.")
    return pages
