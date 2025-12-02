"""Main SLM Builder API."""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from slm_builder.config import SLMConfig, get_recipe_defaults, merge_configs
from slm_builder.data import load_dataset
from slm_builder.data.annotator import annotate_dataset
from slm_builder.data.transforms import create_default_pipeline
from slm_builder.models.base import ModelFactory
from slm_builder.models.export import create_model_bundle
from slm_builder.models.trainer import Trainer, prepare_dataset_for_training
from slm_builder.utils import (
    compute_data_hash,
    detect_hardware,
    get_device_string,
    get_logger,
    recommend_base_models,
    recommend_batch_size,
    recommend_recipe,
    save_json,
    save_jsonl,
    save_metadata,
    scan_dataset_for_pii,
    setup_logging,
    validate_directory_exists,
)

logger = get_logger(__name__)


class SLMBuilder:
    """Main class for building Small/Specialized Language Models."""

    def __init__(
        self,
        project_name: str,
        base_model: str = "gpt2",
        device: Optional[str] = None,
        work_dir: str = "./slm_workdir",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize SLM Builder.

        Args:
            project_name: Name of the project
            base_model: Base model name or path
            device: Device to use ('cpu', 'cuda', 'auto', or None for auto)
            work_dir: Working directory for outputs
            config: Optional configuration dictionary
        """
        # Setup logging
        setup_logging()

        self.project_name = project_name
        self.work_dir = Path(work_dir) / project_name
        self.work_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing SLM Builder", project=project_name, work_dir=str(self.work_dir))

        # Detect hardware
        self.hw_profile = detect_hardware()
        logger.info("Hardware detected", **self.hw_profile)

        # Determine device
        if device is None or device == "auto":
            device = "cuda" if self.hw_profile["has_cuda"] else "cpu"

        # Build configuration
        config_dict = {
            "project_name": project_name,
            "base_model": base_model,
            "device": device,
            "work_dir": str(self.work_dir),
        }

        if config:
            config_dict = merge_configs(config_dict, config)

        self.config = SLMConfig(**config_dict)

        # Preprocessors
        self._preprocessors: List[Callable] = []
        self._postprocessors: List[Callable] = []

        logger.info("SLM Builder initialized", config=self.config.model_dump())

    def register_preprocessor(self, fn: Callable[[List[Dict]], List[Dict]]) -> None:
        """Register a custom preprocessor function.

        Args:
            fn: Function that takes and returns list of records
        """
        self._preprocessors.append(fn)
        logger.info("Registered preprocessor", fn=fn.__name__)

    def register_postprocessor(self, fn: Callable[[List[Dict]], List[Dict]]) -> None:
        """Register a custom postprocessor function.

        Args:
            fn: Function that takes and returns list of records
        """
        self._postprocessors.append(fn)
        logger.info("Registered postprocessor", fn=fn.__name__)

    def build_from_csv(
        self,
        path: str,
        task: str = "qa",
        recipe: str = "lora",
        output_dir: Optional[str] = None,
        annotation_opts: Optional[Dict] = None,
        overrides: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Build SLM from CSV file.

        Args:
            path: Path to CSV file
            task: Task type
            recipe: Training recipe
            output_dir: Optional output directory
            annotation_opts: Optional annotation options
            overrides: Configuration overrides
            **kwargs: Additional arguments for loader

        Returns:
            Build results dictionary
        """
        self.config.task = task
        self.config.recipe = recipe

        if overrides:
            # Apply recipe defaults first
            recipe_defaults = get_recipe_defaults(recipe)
            config_dict = self.config.model_dump()
            config_dict = merge_configs(config_dict, recipe_defaults)
            config_dict = merge_configs(config_dict, overrides)
            self.config = SLMConfig(**config_dict)

        return self._build_from_source(
            source=path,
            source_type="csv",
            output_dir=output_dir,
            annotation_opts=annotation_opts,
            **kwargs,
        )

    def build_from_jsonl(
        self,
        path: str,
        task: str = "qa",
        recipe: str = "lora",
        output_dir: Optional[str] = None,
        overrides: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Build SLM from JSONL file."""
        self.config.task = task
        self.config.recipe = recipe

        if overrides:
            recipe_defaults = get_recipe_defaults(recipe)
            config_dict = merge_configs(self.config.model_dump(), recipe_defaults)
            config_dict = merge_configs(config_dict, overrides)
            self.config = SLMConfig(**config_dict)

        return self._build_from_source(
            source=path, source_type="jsonl", output_dir=output_dir, **kwargs
        )

    def build_from_text_dir(
        self,
        path: str,
        task: str = "generation",
        recipe: str = "lora",
        output_dir: Optional[str] = None,
        overrides: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Build SLM from directory of text files."""
        self.config.task = task
        self.config.recipe = recipe

        if overrides:
            recipe_defaults = get_recipe_defaults(recipe)
            config_dict = merge_configs(self.config.model_dump(), recipe_defaults)
            config_dict = merge_configs(config_dict, overrides)
            self.config = SLMConfig(**config_dict)

        return self._build_from_source(
            source=path, source_type="text_dir", output_dir=output_dir, **kwargs
        )

    def build_from_dataset(
        self,
        records: List[Dict[str, Any]],
        task: str = "qa",
        recipe: str = "lora",
        output_dir: Optional[str] = None,
        overrides: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Build SLM from pre-loaded dataset.

        Args:
            records: List of records in canonical format
            task: Task type
            recipe: Training recipe
            output_dir: Optional output directory
            overrides: Configuration overrides

        Returns:
            Build results
        """
        self.config.task = task
        self.config.recipe = recipe

        if overrides:
            recipe_defaults = get_recipe_defaults(recipe)
            config_dict = merge_configs(self.config.model_dump(), recipe_defaults)
            config_dict = merge_configs(config_dict, overrides)
            self.config = SLMConfig(**config_dict)

        return self._build_pipeline(records, output_dir=output_dir)

    def _build_from_source(
        self,
        source: str,
        source_type: str,
        output_dir: Optional[str] = None,
        annotation_opts: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Internal method to build from a source."""
        logger.info("Loading dataset", source=source, type=source_type, task=self.config.task)

        # Load data
        records = load_dataset(source, task=self.config.task, **kwargs)
        logger.info("Dataset loaded", count=len(records))

        # Optional annotation
        if annotation_opts and annotation_opts.get("launch_ui"):
            logger.info("Launching annotation UI")
            annotated_path = annotate_dataset(
                records,
                task=self.config.task,
                output_path=str(self.work_dir / "annotated.jsonl"),
                **annotation_opts,
            )
            # Reload annotated data
            from slm_builder.utils.serialization import load_jsonl

            records = load_jsonl(annotated_path)

        return self._build_pipeline(records, output_dir=output_dir)

    def _build_pipeline(
        self,
        records: List[Dict[str, Any]],
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the full build pipeline."""
        if output_dir is None:
            output_dir = str(self.work_dir / "output")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Starting build pipeline", records=len(records))

        # Security: PII check
        if not self.config.allow_pii:
            scan_dataset_for_pii(records, allow_pii=False)

        # Save raw data
        raw_data_path = output_path / "raw_data.jsonl"
        save_jsonl(records, str(raw_data_path))

        # Apply custom preprocessors
        for preprocessor in self._preprocessors:
            logger.info("Applying preprocessor", fn=preprocessor.__name__)
            records = preprocessor(records)

        # Load model and tokenizer
        logger.info("Loading model", model=self.config.base_model, device=self.config.device)
        model, tokenizer = ModelFactory.load_model_and_tokenizer(
            self.config.base_model,
            device=self.config.device,
            use_auth_token=self.config.use_auth_token,
            trust_remote_code=self.config.trust_remote_code,
        )

        # Create preprocessing pipeline
        pipeline = create_default_pipeline(self.config.preprocess, tokenizer=tokenizer)
        records = pipeline.apply(records)
        logger.info("Preprocessing complete", records=len(records))

        # Save processed data
        processed_data_path = output_path / "processed_data.jsonl"
        save_jsonl(records, str(processed_data_path))

        # Prepare training dataset
        train_dataset = prepare_dataset_for_training(
            records,
            tokenizer,
            max_length=self.config.preprocess.max_tokens_per_chunk,
        )

        # Train model
        trainer = Trainer(
            config=self.config,
            model=model,
            tokenizer=tokenizer,
            output_dir=str(output_path / "checkpoints"),
        )

        training_results = trainer.train(train_dataset)

        # Apply custom postprocessors
        for postprocessor in self._postprocessors:
            logger.info("Applying postprocessor", fn=postprocessor.__name__)
            records = postprocessor(records)

        # Save metadata
        metadata = {
            "project_name": self.project_name,
            "base_model": self.config.base_model,
            "task": self.config.task,
            "recipe": self.config.recipe,
            "hardware": self.hw_profile,
            "config": self.config.model_dump(),
            "dataset_hash": compute_data_hash(records),
            "num_records": len(records),
            "training_results": training_results,
        }
        save_metadata(metadata, str(output_path))

        result = {
            "model_dir": training_results["model_dir"],
            "output_dir": str(output_path),
            "metrics": training_results,
            "recipe": self.config.recipe,
            "artifact_paths": {
                "raw_data": str(raw_data_path),
                "processed_data": str(processed_data_path),
                "metadata": str(output_path / "metadata.json"),
            },
        }

        logger.info("Build complete", result=result)
        return result

    def export(
        self,
        model_dir: str,
        format: str = "onnx",
        optimize_for: str = "cpu",
        output_dir: Optional[str] = None,
    ) -> str:
        """Export trained model to specified format.

        Args:
            model_dir: Directory with trained model
            format: Export format (onnx, torchscript, huggingface)
            optimize_for: Optimization target (cpu, cuda)
            output_dir: Optional output directory

        Returns:
            Path to exported model
        """
        if output_dir is None:
            output_dir = str(self.work_dir / "exported")

        logger.info("Exporting model", format=format, optimize_for=optimize_for)

        exported_path = create_model_bundle(
            model_dir=model_dir,
            output_dir=output_dir,
            format=format,
            optimize_for=optimize_for,
            quantize=self.config.export.quantize,
            merge_lora=self.config.export.merge_lora,
        )

        logger.info("Export complete", path=exported_path)
        return exported_path

    def serve(self, model_dir: str, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Start serving the model via FastAPI.

        Args:
            model_dir: Directory with model to serve
            host: Host address
            port: Port number
        """
        from slm_builder.serve.fastapi_server import start_server

        logger.info("Starting server", host=host, port=port, model=model_dir)
        start_server(model_dir, host=host, port=port)
