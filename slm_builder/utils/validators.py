"""Input validation utilities."""

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from slm_builder.utils.logging import get_logger

logger = get_logger(__name__)


# PII detection patterns (basic heuristics)
PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
}


def validate_file_exists(path: str) -> Path:
    """Validate that a file exists.

    Args:
        path: File path to validate

    Returns:
        Path object

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    return file_path


def validate_directory_exists(path: str, create: bool = False) -> Path:
    """Validate that a directory exists or create it.

    Args:
        path: Directory path to validate
        create: Whether to create directory if it doesn't exist

    Returns:
        Path object

    Raises:
        FileNotFoundError: If directory doesn't exist and create=False
    """
    dir_path = Path(path)
    if not dir_path.exists():
        if create:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info("Created directory", path=str(dir_path))
        else:
            raise FileNotFoundError(f"Directory not found: {path}")
    elif not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    return dir_path


def detect_pii(text: str, patterns: Optional[Dict[str, re.Pattern]] = None) -> List[Dict[str, Any]]:
    """Detect potential PII in text using regex patterns.

    Args:
        text: Text to scan for PII
        patterns: Optional custom PII patterns (uses defaults if None)

    Returns:
        List of detected PII instances with type and matched text
    """
    if patterns is None:
        patterns = PII_PATTERNS

    detections = []
    for pii_type, pattern in patterns.items():
        matches = pattern.finditer(text)
        for match in matches:
            detections.append(
                {
                    "type": pii_type,
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    return detections


def scan_dataset_for_pii(records: List[Dict[str, Any]], allow_pii: bool = False) -> Dict[str, Any]:
    """Scan a dataset for PII and return report.

    Args:
        records: List of dataset records
        allow_pii: If False, raise error if PII detected

    Returns:
        Dictionary with PII scan results

    Raises:
        ValueError: If PII detected and allow_pii=False
    """
    total_records = len(records)
    records_with_pii = 0
    pii_detections = []

    for i, record in enumerate(records):
        text = record.get("text", "")
        if isinstance(text, str):
            detections = detect_pii(text)
            if detections:
                records_with_pii += 1
                pii_detections.extend(
                    [{**d, "record_index": i} for d in detections[:5]]  # Limit samples
                )

    report = {
        "total_records": total_records,
        "records_with_pii": records_with_pii,
        "pii_percentage": round(100 * records_with_pii / max(total_records, 1), 2),
        "sample_detections": pii_detections[:20],  # Show max 20 samples
    }

    if records_with_pii > 0:
        logger.warning(
            "PII detected in dataset",
            records_with_pii=records_with_pii,
            percentage=report["pii_percentage"],
        )

        if not allow_pii:
            raise ValueError(
                f"PII detected in {records_with_pii} records ({report['pii_percentage']}%). "
                "Set allow_pii=True to proceed or clean the data first."
            )

    return report


def validate_dataset_schema(records: List[Dict[str, Any]], required_fields: List[str]) -> None:
    """Validate that dataset records have required fields.

    Args:
        records: List of dataset records
        required_fields: List of required field names

    Raises:
        ValueError: If any record is missing required fields
    """
    if not records:
        raise ValueError("Dataset is empty")

    missing_fields_samples = []
    for i, record in enumerate(records[:100]):  # Check first 100
        missing = [f for f in required_fields if f not in record]
        if missing:
            missing_fields_samples.append(
                {
                    "record_index": i,
                    "missing_fields": missing,
                }
            )

    if missing_fields_samples:
        raise ValueError(f"Dataset records missing required fields: {missing_fields_samples[:5]}")


def compute_data_hash(records: List[Dict[str, Any]]) -> str:
    """Compute a hash of the dataset for provenance tracking.

    Args:
        records: List of dataset records

    Returns:
        SHA256 hash of dataset
    """
    # Create deterministic string representation
    data_str = str(sorted([str(sorted(r.items())) for r in records]))
    return hashlib.sha256(data_str.encode()).hexdigest()


def validate_column_mapping(
    columns: List[str], mapping: Dict[str, str], required_mappings: List[str]
) -> None:
    """Validate that column mapping contains required keys and valid columns.

    Args:
        columns: Available column names
        mapping: Column name mapping
        required_mappings: Required mapping keys

    Raises:
        ValueError: If mapping is invalid
    """
    # Check required mappings exist
    missing = [k for k in required_mappings if k not in mapping]
    if missing:
        raise ValueError(f"Column mapping missing required keys: {missing}")

    # Check mapped columns exist
    invalid = [v for v in mapping.values() if v not in columns]
    if invalid:
        raise ValueError(f"Mapped columns not found in data: {invalid}. Available: {columns}")


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing/replacing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
    sanitized = sanitized.strip(". ")

    # Limit length
    if len(sanitized) > 200:
        name, ext = sanitized.rsplit(".", 1) if "." in sanitized else (sanitized, "")
        sanitized = name[: 200 - len(ext) - 1] + (f".{ext}" if ext else "")

    return sanitized or "unnamed"


def validate_model_name(model_name: str) -> str:
    """Validate and normalize model name.

    Args:
        model_name: Model name or path

    Returns:
        Validated model name

    Raises:
        ValueError: If model name is invalid
    """
    if not model_name or not model_name.strip():
        raise ValueError("Model name cannot be empty")

    # Check if it's a local path
    if Path(model_name).exists():
        return str(Path(model_name).resolve())

    # Assume it's a HuggingFace model ID
    # Basic validation: should be org/model or just model
    if "/" in model_name:
        parts = model_name.split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"Invalid model name format: {model_name}")

    return model_name
