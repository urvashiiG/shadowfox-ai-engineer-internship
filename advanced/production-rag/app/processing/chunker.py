from hashlib import sha1
from app.processing.cleaner import clean_text

def chunk_pages(pages: list[dict], *, document_id: str, document_name: str, source_type: str, size: int = 800, overlap: int = 150) -> list[dict]:
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("Chunk size must be positive and overlap must be smaller than chunk size.")
    chunks, seen, index = [], set(), 0
    for page in pages:
        text = clean_text(page.get("text", ""))
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                boundary = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
                if boundary > start + size // 2:
                    end = boundary
            body = text[start:end].strip()
            digest = sha1(body.encode("utf-8")).hexdigest()
            if body and digest not in seen:
                seen.add(digest)
                chunks.append({"document_id": document_id, "document_name": document_name, "source_type": source_type,
                               "page_number": page.get("page_number"), "chunk_id": f"{document_id}-{index}",
                               "chunk_index": index, "source_text": body})
                index += 1
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks
