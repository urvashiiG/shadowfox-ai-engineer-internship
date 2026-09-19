"""
tests/test_validators.py

Unit tests for validators.py.

Run with:
    pytest
"""

import os
import sys

# Allow running `pytest` from the project root without needing to install
# the project as a package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import validators


# --------------------------------------------------------------------------
# is_empty_or_whitespace
# --------------------------------------------------------------------------

def test_is_empty_or_whitespace_none():
    assert validators.is_empty_or_whitespace(None) is True


def test_is_empty_or_whitespace_empty_string():
    assert validators.is_empty_or_whitespace("") is True


def test_is_empty_or_whitespace_whitespace_only():
    assert validators.is_empty_or_whitespace("   \n\t  ") is True


def test_is_empty_or_whitespace_valid_text():
    assert validators.is_empty_or_whitespace("Binary search") is False


# --------------------------------------------------------------------------
# validate_text_input (generic) / validate_concept_input
# --------------------------------------------------------------------------

def test_validate_text_input_empty_text():
    is_valid, message = validators.validate_concept_input("")
    assert is_valid is False
    assert "enter some" in message.lower()


def test_validate_text_input_whitespace_only():
    is_valid, message = validators.validate_concept_input("     ")
    assert is_valid is False
    assert message != ""


def test_validate_text_input_valid_text():
    is_valid, message = validators.validate_concept_input("Binary search algorithm")
    assert is_valid is True
    assert message == ""


def test_validate_text_input_too_short():
    is_valid, message = validators.validate_concept_input("ab")
    assert is_valid is False
    assert "too short" in message.lower()


def test_validate_text_input_excessively_long():
    long_text = "a" * (validators.MAX_CONCEPT_LENGTH + 1)
    is_valid, message = validators.validate_concept_input(long_text)
    assert is_valid is False
    assert "too long" in message.lower()


def test_validate_text_input_boundary_exact_max_length():
    boundary_text = "a" * validators.MAX_CONCEPT_LENGTH
    is_valid, message = validators.validate_concept_input(boundary_text)
    assert is_valid is True
    assert message == ""


def test_validate_text_input_boundary_min_length():
    boundary_text = "a" * validators.MIN_TEXT_LENGTH
    is_valid, message = validators.validate_concept_input(boundary_text)
    assert is_valid is True


# --------------------------------------------------------------------------
# validate_notes_input / validate_quiz_topic_input / validate_answer_input
# --------------------------------------------------------------------------

def test_validate_notes_input_valid():
    is_valid, message = validators.validate_notes_input("Photosynthesis converts light into energy.")
    assert is_valid is True
    assert message == ""


def test_validate_notes_input_too_long():
    long_text = "a" * (validators.MAX_NOTES_LENGTH + 1)
    is_valid, message = validators.validate_notes_input(long_text)
    assert is_valid is False


def test_validate_quiz_topic_input_empty():
    is_valid, message = validators.validate_quiz_topic_input("")
    assert is_valid is False


def test_validate_answer_input_valid():
    is_valid, message = validators.validate_answer_input("The mitochondria is the powerhouse of the cell.")
    assert is_valid is True


# --------------------------------------------------------------------------
# validate_quiz_count
# --------------------------------------------------------------------------

def test_validate_quiz_count_valid_5():
    is_valid, message = validators.validate_quiz_count(5)
    assert is_valid is True
    assert message == ""


def test_validate_quiz_count_valid_10():
    is_valid, message = validators.validate_quiz_count(10)
    assert is_valid is True


def test_validate_quiz_count_valid_15():
    is_valid, message = validators.validate_quiz_count(15)
    assert is_valid is True


def test_validate_quiz_count_invalid_number():
    is_valid, message = validators.validate_quiz_count(7)
    assert is_valid is False
    assert message != ""


def test_validate_quiz_count_zero():
    is_valid, message = validators.validate_quiz_count(0)
    assert is_valid is False


def test_validate_quiz_count_negative():
    is_valid, message = validators.validate_quiz_count(-5)
    assert is_valid is False


def test_validate_quiz_count_non_numeric_string():
    is_valid, message = validators.validate_quiz_count("abc")
    assert is_valid is False


def test_validate_quiz_count_string_number_valid():
    is_valid, message = validators.validate_quiz_count("10")
    assert is_valid is True


def test_validate_quiz_count_none():
    is_valid, message = validators.validate_quiz_count(None)
    assert is_valid is False
