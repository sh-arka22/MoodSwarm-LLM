"""Shared pytest fixtures for MoodSwarm tests."""

import os

import pytest

# Disable Opik tracking during tests to avoid noisy 401 warnings
os.environ["OPIK_TRACK_DISABLE"] = "true"


@pytest.fixture
def sample_query_text():
    return "How do RAG systems work?"


@pytest.fixture
def sample_context():
    return "RAG (Retrieval-Augmented Generation) combines retrieval with generation to produce grounded answers."


@pytest.fixture
def sample_dirty_text():
    return "Hello!!!  @#$  This is    a test   **document** with    extra spaces."
