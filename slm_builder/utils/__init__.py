"""Utils package initialization."""

from slm_builder.utils.hw import (
    detect_hardware,
    estimate_training_time,
    get_device_string,
    recommend_base_models,
    recommend_batch_size,
    recommend_recipe,
)
from slm_builder.utils.logging import get_logger, setup_logging, with_context
from slm_builder.utils.serialization import (
    load_json,
    load_jsonl,
    load_metadata,
    load_yaml,
    save_json,
    save_jsonl,
    save_metadata,
    save_yaml,
)
from slm_builder.utils.validators import (
    compute_data_hash,
    detect_pii,
    sanitize_filename,
    scan_dataset_for_pii,
    validate_column_mapping,
    validate_dataset_schema,
    validate_directory_exists,
    validate_file_exists,
    validate_model_name,
)

__all__ = [
    # Hardware
    "detect_hardware",
    "recommend_base_models",
    "recommend_recipe",
    "recommend_batch_size",
    "estimate_training_time",
    "get_device_string",
    # Logging
    "get_logger",
    "setup_logging",
    "with_context",
    # Serialization
    "save_json",
    "load_json",
    "save_yaml",
    "load_yaml",
    "save_jsonl",
    "load_jsonl",
    "save_metadata",
    "load_metadata",
    # Validators
    "validate_file_exists",
    "validate_directory_exists",
    "detect_pii",
    "scan_dataset_for_pii",
    "validate_dataset_schema",
    "compute_data_hash",
    "validate_column_mapping",
    "sanitize_filename",
    "validate_model_name",
]
