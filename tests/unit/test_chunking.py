"""Unit tests for text cleaning and chunking operations."""

import hashlib
import uuid

from llm_engineering.application.preprocessing.operations.cleaning import clean_text


class TestCleanText:
    def test_strips_special_characters(self):
        result = clean_text("Hello @#$ world!")
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result

    def test_collapses_whitespace(self):
        result = clean_text("hello    world")
        assert result == "hello world"

    def test_strips_leading_trailing_whitespace(self):
        result = clean_text("  hello world  ")
        assert result == "hello world"

    def test_preserves_alphanumeric_and_punctuation(self):
        result = clean_text("Hello, world. Test 123!")
        assert "Hello" in result
        assert "world" in result
        assert "123" in result

    def test_empty_string(self):
        result = clean_text("")
        assert result == ""


class TestChunkDeterministicIds:
    """Verify that same content produces same UUID (MD5-based deterministic IDs)."""

    def test_same_content_same_uuid(self):
        content = "This is a test chunk of text."
        id1 = uuid.UUID(hashlib.md5(content.encode()).hexdigest())
        id2 = uuid.UUID(hashlib.md5(content.encode()).hexdigest())
        assert id1 == id2

    def test_different_content_different_uuid(self):
        id1 = uuid.UUID(hashlib.md5("chunk one".encode()).hexdigest())
        id2 = uuid.UUID(hashlib.md5("chunk two".encode()).hexdigest())
        assert id1 != id2
