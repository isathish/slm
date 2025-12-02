"""Base model adapter and factory."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from slm_builder.utils import get_logger, validate_model_name

logger = get_logger(__name__)


class ModelFactory:
    """Factory for loading and creating models."""

    @staticmethod
    def load_model_and_tokenizer(
        model_name: str,
        device: str = "cpu",
        use_auth_token: Optional[str] = None,
        trust_remote_code: bool = False,
        load_in_8bit: bool = False,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """Load model and tokenizer from HuggingFace or local path.

        Args:
            model_name: Model name or path
            device: Device to load on
            use_auth_token: Optional auth token for private models
            trust_remote_code: Whether to trust remote code
            load_in_8bit: Load model in 8-bit (requires bitsandbytes)
            **kwargs: Additional model loading arguments

        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers not installed. Install with: pip install slm-builder[full]"
            )

        model_name = validate_model_name(model_name)
        logger.info("Loading model", model=model_name, device=device)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=use_auth_token,
            trust_remote_code=trust_remote_code,
        )

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model_kwargs = {
            "use_auth_token": use_auth_token,
            "trust_remote_code": trust_remote_code,
            **kwargs,
        }

        if load_in_8bit:
            try:
                model_kwargs["load_in_8bit"] = True
                model_kwargs["device_map"] = "auto"
            except Exception as e:
                logger.warning("8-bit loading failed, falling back to normal", error=str(e))
                model_kwargs.pop("load_in_8bit", None)
                model_kwargs.pop("device_map", None)

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Move to device if not using device_map
        if "device_map" not in model_kwargs:
            model = model.to(device)

        logger.info("Model loaded", params=sum(p.numel() for p in model.parameters()))

        return model, tokenizer

    @staticmethod
    def save_model_and_tokenizer(model: Any, tokenizer: Any, output_dir: str, **kwargs) -> None:
        """Save model and tokenizer to directory.

        Args:
            model: Model to save
            tokenizer: Tokenizer to save
            output_dir: Output directory
            **kwargs: Additional save arguments
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Saving model", path=output_dir)

        # Save model
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(output_dir, **kwargs)
        else:
            # Handle PEFT models
            if hasattr(model, "base_model"):
                model.base_model.save_pretrained(output_dir, **kwargs)

        # Save tokenizer
        tokenizer.save_pretrained(output_dir)

        logger.info("Model saved", path=output_dir)

    @staticmethod
    def get_model_info(model: Any) -> Dict[str, Any]:
        """Get information about a model.

        Args:
            model: Model instance

        Returns:
            Dictionary with model information
        """
        info = {
            "model_type": type(model).__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }

        # Add config info if available
        if hasattr(model, "config"):
            config = model.config
            info.update(
                {
                    "vocab_size": getattr(config, "vocab_size", None),
                    "hidden_size": getattr(config, "hidden_size", None),
                    "num_layers": getattr(config, "num_hidden_layers", None),
                    "num_attention_heads": getattr(config, "num_attention_heads", None),
                }
            )

        return info


def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cpu",
    **kwargs,
) -> str:
    """Generate text from a prompt.

    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        prompt: Input prompt
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        device: Device
        **kwargs: Additional generation arguments

    Returns:
        Generated text
    """
    import torch

    model.eval()

    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            **kwargs,
        )

    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt from output
    if generated.startswith(prompt):
        generated = generated[len(prompt) :].strip()

    return generated
