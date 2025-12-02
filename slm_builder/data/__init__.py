"""Data package initialization."""

from slm_builder.data.loaders import (
    CSVLoader,
    JSONLLoader,
    TextDirLoader,
    URLLoader,
    get_loader,
    load_dataset,
)
from slm_builder.data.schemas import (
    DatasetRecord,
    create_classification_record,
    create_instruction_record,
    create_qa_record,
    record_to_dict,
    validate_record,
)
from slm_builder.data.transforms import (
    ChunkLongTexts,
    ConvertToInstructionFormat,
    Deduplicate,
    FilterByLength,
    NormalizeText,
    Pipeline,
    TokenizeRecords,
    create_default_pipeline,
)

__all__ = [
    # Loaders
    "CSVLoader",
    "JSONLLoader",
    "TextDirLoader",
    "URLLoader",
    "get_loader",
    "load_dataset",
    # Schemas
    "DatasetRecord",
    "create_qa_record",
    "create_classification_record",
    "create_instruction_record",
    "record_to_dict",
    "validate_record",
    # Transforms
    "NormalizeText",
    "Deduplicate",
    "ChunkLongTexts",
    "FilterByLength",
    "TokenizeRecords",
    "ConvertToInstructionFormat",
    "Pipeline",
    "create_default_pipeline",
]
