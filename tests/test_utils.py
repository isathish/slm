"""Tests for utilities."""

import pytest

from slm_builder.utils.validators import (
    detect_pii,
    sanitize_filename,
    validate_model_name,
)


def test_detect_pii():
    """Test PII detection."""
    text = "Contact me at john@example.com or call 555-123-4567"

    detections = detect_pii(text)

    # Should detect email and phone
    assert len(detections) > 0
    types = [d["type"] for d in detections]
    assert "email" in types
    assert "phone" in types


def test_sanitize_filename():
    """Test filename sanitization."""
    # Test invalid characters
    assert sanitize_filename("file<>:name") == "file___name"

    # Test spaces and dots
    assert sanitize_filename("  file.name  ") == "file.name"

    # Test long names
    long_name = "a" * 250
    sanitized = sanitize_filename(long_name)
    assert len(sanitized) <= 200


def test_validate_model_name():
    """Test model name validation."""
    # Valid HF model name
    assert validate_model_name("gpt2") == "gpt2"
    assert validate_model_name("bert-base-uncased") == "bert-base-uncased"
    assert validate_model_name("openai/gpt2") == "openai/gpt2"

    # Invalid format should raise
    with pytest.raises(ValueError):
        validate_model_name("org/model/extra")

    with pytest.raises(ValueError):
        validate_model_name("")


if __name__ == "__main__":
    pytest.main([__file__])
