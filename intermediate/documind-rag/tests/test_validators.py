"""
test_validators.py

Unit tests for validators.py.

These tests do NOT require an API key and make NO network calls.
"""

import sys
import os
import io

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from validators import validate_query, validate_uploaded_files, validate_extracted_text


class FakeUploadedFile:
    """Minimal stand-in for a Streamlit UploadedFile, for testing."""

    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content
        self.size = len(content)

    def getbuffer(self):
        return self._content


def test_empty_query_rejected():
    is_valid, error = validate_query("")
    assert is_valid is False
    assert error is not None


def test_whitespace_only_query_rejected():
    is_valid, error = validate_query("    ")
    assert is_valid is False


def test_valid_query_accepted():
    is_valid, error = validate_query("What is database normalization?")
    assert is_valid is True
    assert error is None


def test_too_long_query_rejected():
    is_valid, error = validate_query("a" * 5000)
    assert is_valid is False


def test_unsupported_file_rejected():
    fake_file = FakeUploadedFile("notes.docx", b"some content")
    is_valid, error = validate_uploaded_files([fake_file])
    assert is_valid is False
    assert "Unsupported" in error


def test_empty_file_rejected():
    fake_file = FakeUploadedFile("notes.txt", b"")
    is_valid, error = validate_uploaded_files([fake_file])
    assert is_valid is False
    assert "empty" in error.lower()


def test_no_files_rejected():
    is_valid, error = validate_uploaded_files([])
    assert is_valid is False


def test_valid_txt_file_accepted():
    fake_file = FakeUploadedFile("notes.txt", b"Some real study notes content.")
    is_valid, error = validate_uploaded_files([fake_file])
    assert is_valid is True
    assert error is None


def test_valid_pdf_file_accepted():
    fake_file = FakeUploadedFile("notes.pdf", b"%PDF-1.4 fake pdf bytes")
    is_valid, error = validate_uploaded_files([fake_file])
    assert is_valid is True


def test_empty_extracted_text_rejected():
    is_valid, error = validate_extracted_text("   ", "notes.pdf")
    assert is_valid is False


def test_nonempty_extracted_text_accepted():
    is_valid, error = validate_extracted_text("Some extracted text.", "notes.pdf")
    assert is_valid is True
