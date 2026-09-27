"""
test_chunking.py

Unit tests for the chunk_text function in document_processor.py.

These tests do NOT require an API key and make NO network calls.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from document_processor import chunk_text


def test_empty_text_produces_no_chunks():
    chunks = chunk_text([], source="empty.txt")
    assert chunks == []


def test_empty_page_text_produces_no_chunks():
    pages = [{"page": None, "text": "   "}]
    chunks = chunk_text(pages, source="blank.txt")
    assert chunks == []


def test_short_text_produces_single_chunk():
    pages = [{"page": None, "text": "This is a short piece of text."}]
    chunks = chunk_text(pages, source="short.txt", chunk_size=800, chunk_overlap=150)
    assert len(chunks) == 1
    assert chunks[0]["source"] == "short.txt"
    assert chunks[0]["chunk_id"] == 0


def test_long_text_splits_into_multiple_chunks():
    long_text = "word " * 1000  # long enough to require multiple chunks
    pages = [{"page": None, "text": long_text}]
    chunks = chunk_text(pages, source="long.txt", chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1


def test_chunk_overlap_behaves_correctly():
    long_text = "abcdefghij " * 200
    pages = [{"page": None, "text": long_text}]
    chunks = chunk_text(pages, source="overlap.txt", chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    # Consecutive chunks should share some overlapping content.
    first_tail = chunks[0]["text"][-30:]
    second_text = chunks[1]["text"]
    overlap_found = any(first_tail[i:i + 10] in second_text for i in range(0, len(first_tail) - 10))
    assert overlap_found


def test_no_empty_chunks_produced():
    long_text = "a" * 5000
    pages = [{"page": None, "text": long_text}]
    chunks = chunk_text(pages, source="dense.txt", chunk_size=300, chunk_overlap=50)
    for chunk in chunks:
        assert chunk["text"].strip() != ""


def test_chunk_ids_are_sequential():
    long_text = "sentence number " * 300
    pages = [{"page": None, "text": long_text}]
    chunks = chunk_text(pages, source="seq.txt", chunk_size=100, chunk_overlap=20)
    ids = [c["chunk_id"] for c in chunks]
    assert ids == list(range(len(chunks)))


def test_page_metadata_preserved_for_pdf_like_input():
    pages = [
        {"page": 1, "text": "Content of page one."},
        {"page": 2, "text": "Content of page two."},
    ]
    chunks = chunk_text(pages, source="doc.pdf", chunk_size=800, chunk_overlap=150)
    pages_seen = {c["page"] for c in chunks}
    assert pages_seen == {1, 2}


def test_txt_page_is_none():
    pages = [{"page": None, "text": "Some plain text content."}]
    chunks = chunk_text(pages, source="notes.txt")
    assert all(c["page"] is None for c in chunks)
