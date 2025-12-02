"""Training orchestration and recipes."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from slm_builder.config import SLMConfig
from slm_builder.models.base import ModelFactory
from slm_builder.models.peft_utils import apply_lora, get_peft_model_info, merge_lora_adapters
from slm_builder.utils import get_logger, save_json, save_metadata

logger = get_logger(__name__)


class Trainer:
    """Model trainer with recipe support."""

    def __init__(
        self,
        config: SLMConfig,
        model: Any,
        tokenizer: Any,
        output_dir: str,
    ):
        """Initialize trainer.

        Args:
            config: SLM configuration
            model: Model to train
            tokenizer: Tokenizer
            output_dir: Output directory for checkpoints
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self, train_dataset: Any, eval_dataset: Optional[Any] = None, **kwargs
    ) -> Dict[str, Any]:
        """Train model using specified recipe.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            **kwargs: Additional training arguments

        Returns:
            Training results dictionary
        """
        recipe = self.config.recipe
        logger.info("Starting training", recipe=recipe, output_dir=str(self.output_dir))

        if recipe == "lora":
            return self._train_lora(train_dataset, eval_dataset, **kwargs)
        elif recipe == "finetune":
            return self._train_finetune(train_dataset, eval_dataset, **kwargs)
        elif recipe == "instruction-tune":
            return self._train_instruction(train_dataset, eval_dataset, **kwargs)
        else:
            raise ValueError(f"Unsupported recipe: {recipe}")

    def _train_lora(
        self, train_dataset: Any, eval_dataset: Optional[Any] = None, **kwargs
    ) -> Dict[str, Any]:
        """Train with LoRA recipe."""
        try:
            from transformers import Trainer, TrainingArguments
        except ImportError:
            raise ImportError("transformers required for training")

        logger.info("Applying LoRA")
        self.model = apply_lora(self.model, self.config.lora)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.training.epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_steps=self.config.training.warmup_steps,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            eval_steps=self.config.training.eval_steps if eval_dataset else None,
            save_total_limit=self.config.training.save_total_limit,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            fp16=self.config.training.fp16 and self.config.device == "cuda",
            evaluation_strategy="steps" if eval_dataset else "no",
            load_best_model_at_end=True if eval_dataset else False,
            seed=self.config.training.seed,
            report_to=[],  # Disable default reporting
            **kwargs,
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        # Train
        logger.info("Starting LoRA training")
        train_result = trainer.train()

        # Save model
        best_model_dir = self.output_dir / "best"
        best_model_dir.mkdir(exist_ok=True)

        trainer.save_model(str(best_model_dir))
        self.tokenizer.save_pretrained(str(best_model_dir))

        # Save training info
        results = {
            "recipe": "lora",
            "train_loss": train_result.training_loss,
            "train_samples": len(train_dataset),
            "model_dir": str(best_model_dir),
            "peft_info": get_peft_model_info(self.model),
        }

        if eval_dataset:
            eval_results = trainer.evaluate()
            results["eval_loss"] = eval_results.get("eval_loss")

        save_json(results, str(self.output_dir / "training_results.json"))
        logger.info("Training complete", results=results)

        return results

    def _train_finetune(
        self, train_dataset: Any, eval_dataset: Optional[Any] = None, **kwargs
    ) -> Dict[str, Any]:
        """Train with full fine-tuning recipe."""
        try:
            from transformers import Trainer, TrainingArguments
        except ImportError:
            raise ImportError("transformers required for training")

        # Training arguments (similar to LoRA but without PEFT)
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.training.epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_steps=self.config.training.warmup_steps,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            eval_steps=self.config.training.eval_steps if eval_dataset else None,
            save_total_limit=self.config.training.save_total_limit,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            fp16=self.config.training.fp16 and self.config.device == "cuda",
            evaluation_strategy="steps" if eval_dataset else "no",
            load_best_model_at_end=True if eval_dataset else False,
            seed=self.config.training.seed,
            report_to=[],
            **kwargs,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        logger.info("Starting full fine-tuning")
        train_result = trainer.train()

        # Save
        best_model_dir = self.output_dir / "best"
        best_model_dir.mkdir(exist_ok=True)

        trainer.save_model(str(best_model_dir))
        self.tokenizer.save_pretrained(str(best_model_dir))

        results = {
            "recipe": "finetune",
            "train_loss": train_result.training_loss,
            "train_samples": len(train_dataset),
            "model_dir": str(best_model_dir),
        }

        if eval_dataset:
            eval_results = trainer.evaluate()
            results["eval_loss"] = eval_results.get("eval_loss")

        save_json(results, str(self.output_dir / "training_results.json"))
        logger.info("Training complete", results=results)

        return results

    def _train_instruction(
        self, train_dataset: Any, eval_dataset: Optional[Any] = None, **kwargs
    ) -> Dict[str, Any]:
        """Train with instruction-tuning recipe (uses LoRA by default)."""
        logger.info("Instruction tuning (using LoRA)")
        return self._train_lora(train_dataset, eval_dataset, **kwargs)


def prepare_dataset_for_training(
    records: List[Dict[str, Any]],
    tokenizer: Any,
    max_length: int = 512,
) -> Any:
    """Prepare dataset for training.

    Args:
        records: List of records
        tokenizer: Tokenizer
        max_length: Maximum sequence length

    Returns:
        HuggingFace Dataset object
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("datasets required. Install with: pip install datasets")

    # Extract texts
    texts = [record.get("text", "") for record in records]

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    # Create dataset
    dataset_dict = {"text": texts}
    dataset = Dataset.from_dict(dataset_dict)

    # Tokenize
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Add labels (for causal LM, labels = input_ids)
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    tokenized = tokenized.map(add_labels, batched=True)

    logger.info("Dataset prepared", samples=len(tokenized))
    return tokenized
