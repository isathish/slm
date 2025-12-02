"""Serialization utilities for saving and loading models and data."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict

import yaml

from slm_builder.utils.logging import get_logger

logger = get_logger(__name__)


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Save data to JSON file.

    Args:
        data: Data to save (must be JSON-serializable)
        path: Output file path
        indent: JSON indentation level
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    logger.debug("Saved JSON", path=str(path_obj))


def load_json(path: str) -> Any:
    """Load data from JSON file.

    Args:
        path: Input file path

    Returns:
        Loaded data
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.debug("Loaded JSON", path=path)
    return data


def save_yaml(data: Any, path: str) -> None:
    """Save data to YAML file.

    Args:
        data: Data to save
        path: Output file path
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    logger.debug("Saved YAML", path=str(path_obj))


def load_yaml(path: str) -> Any:
    """Load data from YAML file.

    Args:
        path: Input file path

    Returns:
        Loaded data
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    logger.debug("Loaded YAML", path=path)
    return data


def save_jsonl(records: list, path: str) -> None:
    """Save list of records to JSONL file (one JSON per line).

    Args:
        records: List of records (must be JSON-serializable)
        path: Output file path
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Saved JSONL", path=str(path_obj), count=len(records))


def load_jsonl(path: str, max_records: int = None) -> list:
    """Load records from JSONL file.

    Args:
        path: Input file path
        max_records: Optional limit on number of records to load

    Returns:
        List of records
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_records and i >= max_records:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info("Loaded JSONL", path=path, count=len(records))
    return records


def save_pickle(data: Any, path: str) -> None:
    """Save data using pickle.

    Args:
        data: Data to save
        path: Output file path
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, "wb") as f:
        pickle.dump(data, f)

    logger.debug("Saved pickle", path=str(path_obj))


def load_pickle(path: str) -> Any:
    """Load data from pickle file.

    Args:
        path: Input file path

    Returns:
        Loaded data
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    logger.debug("Loaded pickle", path=path)
    return data


def save_metadata(metadata: Dict[str, Any], work_dir: str, filename: str = "metadata.json") -> str:
    """Save metadata to work directory.

    Args:
        metadata: Metadata dictionary
        work_dir: Working directory
        filename: Metadata filename

    Returns:
        Path to saved metadata file
    """
    path = Path(work_dir) / filename
    save_json(metadata, str(path))
    return str(path)


def load_metadata(work_dir: str, filename: str = "metadata.json") -> Dict[str, Any]:
    """Load metadata from work directory.

    Args:
        work_dir: Working directory
        filename: Metadata filename

    Returns:
        Metadata dictionary
    """
    path = Path(work_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return load_json(str(path))
