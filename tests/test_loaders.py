"""Tests for data loaders."""

import tempfile
from pathlib import Path

import pytest

from slm_builder.data.loaders import CSVLoader, JSONLLoader, load_dataset


def test_csv_loader_qa():
    """Test CSV loader for QA task."""
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("question,answer\n")
        f.write("What is AI?,Artificial Intelligence\n")
        f.write("What is ML?,Machine Learning\n")
        csv_path = f.name

    try:
        loader = CSVLoader(task="qa")
        records = loader.load(csv_path)

        assert len(records) == 2
        assert records[0]["task"] == "qa"
        assert "question" in records[0]["label"]
        assert "answer" in records[0]["label"]
        assert records[0]["label"]["question"] == "What is AI?"
        assert records[0]["label"]["answer"] == "Artificial Intelligence"
    finally:
        Path(csv_path).unlink()


def test_jsonl_loader():
    """Test JSONL loader."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write('{"id": "1", "text": "Hello", "task": "qa"}\n')
        f.write('{"id": "2", "text": "World", "task": "qa"}\n')
        jsonl_path = f.name

    try:
        loader = JSONLLoader(task="qa")
        records = loader.load(jsonl_path)

        assert len(records) == 2
        assert records[0]["id"] == "1"
        assert records[0]["text"] == "Hello"
        assert records[1]["id"] == "2"
    finally:
        Path(jsonl_path).unlink()


def test_load_dataset_auto():
    """Test automatic loader detection."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("question,answer\n")
        f.write("Test Q,Test A\n")
        csv_path = f.name

    try:
        records = load_dataset(csv_path, task="qa")
        assert len(records) == 1
        assert records[0]["task"] == "qa"
    finally:
        Path(csv_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])
