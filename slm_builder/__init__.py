"""
SLM Builder - End-to-end toolkit for building Small/Specialized Language Models.

This package provides tools for:
- Data ingestion from multiple sources (CSV, JSONL, TXT, URLs, databases)
- Data preprocessing and annotation
- Model training with PEFT/LoRA and full fine-tuning
- Model evaluation and export (ONNX, TorchScript)
- Model serving via FastAPI

Example:
    >>> from slm_builder import SLMBuilder
    >>> builder = SLMBuilder(project_name='my-slm')
    >>> result = builder.build_from_csv('data.csv', task='qa', recipe='lora')
"""

__version__ = "1.0.0"
__author__ = "SLM Builder Team"

from slm_builder.api import SLMBuilder

__all__ = ["SLMBuilder", "__version__"]
