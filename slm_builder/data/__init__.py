"""Data package initialization."""

from slm_builder.data.api_loaders import APILoader, load_from_api
from slm_builder.data.database_loaders import (
    MongoDBLoader,
    SQLLoader,
    load_from_mongodb,
    load_from_sql,
)
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
from slm_builder.data.splitting import (
    DatasetSplitter,
    DatasetValidator,
    split_dataset,
    validate_dataset,
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
    # Database loaders
    "SQLLoader",
    "MongoDBLoader",
    "load_from_sql",
    "load_from_mongodb",
    # API loaders
    "APILoader",
    "load_from_api",
    # Schemas
    "DatasetRecord",
    "create_qa_record",
    "create_classification_record",
    "create_instruction_record",
    "record_to_dict",
    "validate_record",
    # Splitting and validation
    "DatasetSplitter",
    "DatasetValidator",
    "split_dataset",
    "validate_dataset",
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
