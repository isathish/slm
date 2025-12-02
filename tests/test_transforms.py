"""Tests for transformations."""

import pytest

from slm_builder.data.transforms import (
    ChunkLongTexts,
    Deduplicate,
    FilterByLength,
    NormalizeText,
)


def test_normalize_text():
    """Test text normalization."""
    records = [
        {"id": "1", "text": "  Hello   World  ", "task": "qa"},
        {"id": "2", "text": "Check http://example.com here", "task": "qa"},
    ]

    transform = NormalizeText(
        lowercase=False,
        strip_urls=True,
        normalize_whitespace=True,
    )

    result = transform.apply(records)

    assert result[0]["text"] == "Hello World"
    assert "http://" not in result[1]["text"]


def test_deduplicate():
    """Test deduplication."""
    records = [
        {"id": "1", "text": "Hello", "task": "qa"},
        {"id": "2", "text": "Hello", "task": "qa"},  # Duplicate
        {"id": "3", "text": "World", "task": "qa"},
    ]

    transform = Deduplicate(key_field="text")
    result = transform.apply(records)

    assert len(result) == 2
    assert result[0]["text"] == "Hello"
    assert result[1]["text"] == "World"


def test_filter_by_length():
    """Test length filtering."""
    records = [
        {"id": "1", "text": "Hi", "task": "qa"},  # Too short
        {"id": "2", "text": "Hello World", "task": "qa"},  # OK
        {"id": "3", "text": "This is a longer text", "task": "qa"},  # OK
    ]

    transform = FilterByLength(min_length=10, max_length=50)
    result = transform.apply(records)

    assert len(result) == 2
    assert all(len(r["text"]) >= 10 for r in result)


def test_chunk_long_texts():
    """Test text chunking."""
    records = [
        {"id": "1", "text": " ".join(["word"] * 100), "task": "qa"},
    ]

    transform = ChunkLongTexts(max_tokens=20, overlap=5)
    result = transform.apply(records)

    # Should create multiple chunks
    assert len(result) > 1
    assert all("chunk" in r["id"] for r in result)
    assert result[0]["metadata"]["chunk_index"] == 0


if __name__ == "__main__":
    pytest.main([__file__])
