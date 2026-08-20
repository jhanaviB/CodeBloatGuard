"""
Shared fixtures.

The API key is set before anything imports config, which reads it at module
scope and raises KeyError without it. Nothing in the offline suite makes a
call, so the value only has to exist.
"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_repo() -> Path:
    """Three files, one duplicate pair across two of them."""
    return FIXTURES / "sample_repo"


@pytest.fixture
def edge_cases() -> Path:
    """The three chunker bugs that indexing psf/requests turned up."""
    return FIXTURES / "edge_cases"
