"""Models package initialization."""

from slm_builder.models.base import ModelFactory, generate_text
from slm_builder.models.comparison import (
    ExperimentTracker,
    ModelComparator,
    compare_models,
)
from slm_builder.models.evaluation import Evaluator, evaluate_model
from slm_builder.models.export import create_model_bundle, export_to_onnx
from slm_builder.models.peft_utils import apply_lora, merge_lora_adapters
from slm_builder.models.registry import (
    ModelRegistry,
    detect_model_source,
    get_registry,
    list_models,
    register_model,
    search_models,
)
from slm_builder.models.trainer import Trainer, prepare_dataset_for_training

__all__ = [
    "ModelFactory",
    "generate_text",
    "Trainer",
    "prepare_dataset_for_training",
    "apply_lora",
    "merge_lora_adapters",
    "export_to_onnx",
    "create_model_bundle",
    "ModelRegistry",
    "get_registry",
    "register_model",
    "detect_model_source",
    "list_models",
    "search_models",
    "Evaluator",
    "evaluate_model",
    "ModelComparator",
    "ExperimentTracker",
    "compare_models",
]
