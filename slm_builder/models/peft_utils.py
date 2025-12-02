"""PEFT/LoRA utilities."""

from typing import Any, Dict, List, Optional

from slm_builder.config import LoRAConfig
from slm_builder.utils import get_logger

logger = get_logger(__name__)


def apply_lora(model: Any, lora_config: LoRAConfig, **kwargs) -> Any:
    """Apply LoRA adapters to a model.

    Args:
        model: Base model
        lora_config: LoRA configuration
        **kwargs: Additional PEFT arguments

    Returns:
        Model with LoRA adapters applied
    """
    try:
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model
    except ImportError:
        raise ImportError("peft not installed. Install with: pip install slm-builder[full]")

    logger.info("Applying LoRA adapters", r=lora_config.r, alpha=lora_config.lora_alpha)

    # Convert our config to PEFT config
    peft_config = PeftLoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        task_type=lora_config.task_type,
        **kwargs,
    )

    # Apply PEFT
    model = get_peft_model(model, peft_config)

    # Log trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable / total

    logger.info(
        "LoRA applied",
        trainable_params=trainable,
        total_params=total,
        percentage=f"{percentage:.2f}%",
    )

    return model


def merge_lora_adapters(model: Any) -> Any:
    """Merge LoRA adapters into base model.

    Args:
        model: PEFT model with LoRA adapters

    Returns:
        Merged model
    """
    logger.info("Merging LoRA adapters")

    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    else:
        logger.warning("Model does not support merging, returning as-is")

    return model


def get_peft_model_info(model: Any) -> Dict[str, Any]:
    """Get information about PEFT model.

    Args:
        model: PEFT model

    Returns:
        Dictionary with PEFT info
    """
    info = {}

    if hasattr(model, "peft_config"):
        config = list(model.peft_config.values())[0] if model.peft_config else None
        if config:
            info["peft_type"] = config.peft_type if hasattr(config, "peft_type") else "unknown"
            info["r"] = getattr(config, "r", None)
            info["lora_alpha"] = getattr(config, "lora_alpha", None)
            info["lora_dropout"] = getattr(config, "lora_dropout", None)
            info["target_modules"] = getattr(config, "target_modules", None)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    info["trainable_parameters"] = trainable
    info["total_parameters"] = total
    info["trainable_percentage"] = 100 * trainable / total if total > 0 else 0

    return info


def prepare_model_for_kbit_training(model: Any) -> Any:
    """Prepare model for k-bit training (8-bit, 4-bit).

    Args:
        model: Model to prepare

    Returns:
        Prepared model
    """
    try:
        from peft import prepare_model_for_kbit_training as peft_prepare

        logger.info("Preparing model for k-bit training")
        return peft_prepare(model)
    except ImportError:
        logger.warning("peft not available, skipping k-bit preparation")
        return model
