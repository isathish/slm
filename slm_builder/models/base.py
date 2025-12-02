"""Base model adapter and factory."""

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from slm_builder.models.registry import ModelSource, detect_model_source
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
        load_in_4bit: bool = False,
        quantization_config: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """Load model and tokenizer from any supported source.

        Supports:
        - HuggingFace Hub models (org/model-name or simple names)
        - Local directories with HuggingFace models
        - Ollama models (prefix with ollama:)
        - GGUF files (local or HTTP/S3 URLs)
        - HTTP/S3 URLs to model files
        - Models from registry zoo

        Args:
            model_name: Model name, path, or URL
            device: Device to load on
            use_auth_token: Optional auth token for private models
            trust_remote_code: Whether to trust remote code
            load_in_8bit: Load model in 8-bit (requires bitsandbytes)
            load_in_4bit: Load model in 4-bit (requires bitsandbytes)
            quantization_config: Custom quantization configuration
            **kwargs: Additional model loading arguments

        Returns:
            Tuple of (model, tokenizer)
        """
        # Detect model source
        source, resolved_path = detect_model_source(model_name)
        logger.info(
            "Loading model",
            original=model_name,
            source=source,
            resolved=resolved_path,
            device=device,
        )

        # Route to appropriate loader
        if source == ModelSource.OLLAMA:
            return ModelFactory._load_from_ollama(resolved_path, device, **kwargs)
        elif source == ModelSource.GGUF_FILE:
            return ModelFactory._load_from_gguf(resolved_path, device, use_auth_token, **kwargs)
        elif source in [ModelSource.HTTP_URL, ModelSource.S3_URL]:
            return ModelFactory._load_from_url(
                resolved_path, device, use_auth_token, trust_remote_code, **kwargs
            )
        else:
            # HuggingFace Hub or local path
            return ModelFactory._load_from_huggingface(
                resolved_path,
                device,
                use_auth_token,
                trust_remote_code,
                load_in_8bit,
                load_in_4bit,
                quantization_config,
                **kwargs,
            )

    @staticmethod
    def _load_from_huggingface(
        model_path: str,
        device: str = "cpu",
        use_auth_token: Optional[str] = None,
        trust_remote_code: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        quantization_config: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """Load model from HuggingFace Hub or local path."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers not installed. Install with: pip install slm-builder[full]"
            )

        model_path = validate_model_name(model_path)
        logger.info("Loading from HuggingFace", path=model_path, device=device)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
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

        # Handle quantization
        if load_in_4bit or load_in_8bit or quantization_config:
            try:
                from transformers import BitsAndBytesConfig

                if quantization_config:
                    bnb_config = BitsAndBytesConfig(**quantization_config)
                elif load_in_4bit:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype="float16",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                else:  # load_in_8bit
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

                model_kwargs["quantization_config"] = bnb_config
                model_kwargs["device_map"] = "auto"
                logger.info("Using quantization", config=bnb_config)
            except Exception as e:
                logger.warning("Quantization failed, falling back to normal loading", error=str(e))

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        # Move to device if not using device_map
        if "device_map" not in model_kwargs:
            model = model.to(device)

        logger.info("Model loaded", params=sum(p.numel() for p in model.parameters()))

        return model, tokenizer

    @staticmethod
    def _load_from_ollama(model_name: str, device: str = "cpu", **kwargs) -> Tuple[Any, Any]:
        """Load model from Ollama.

        Note: This creates a wrapper around Ollama CLI for inference.
        For training, the model will be converted/exported from Ollama.
        """
        logger.info("Loading from Ollama", model=model_name)

        try:
            import subprocess

            # Check if model exists
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if model_name not in result.stdout:
                logger.warning("Model not found in Ollama, attempting to pull", model=model_name)
                subprocess.run(["ollama", "pull", model_name], check=True, timeout=300)

            # Create a wrapper for Ollama
            from slm_builder.models.ollama_wrapper import OllamaModelWrapper

            model = OllamaModelWrapper(model_name, device)
            tokenizer = model.get_tokenizer()

            logger.info("Ollama model loaded", model=model_name)
            return model, tokenizer

        except Exception as e:
            logger.error("Failed to load from Ollama", error=str(e))
            raise RuntimeError(
                f"Failed to load Ollama model '{model_name}'. "
                f"Ensure Ollama is installed and running. Error: {e}"
            )

    @staticmethod
    def _load_from_gguf(
        gguf_path: str, device: str = "cpu", use_auth_token: Optional[str] = None, **kwargs
    ) -> Tuple[Any, Any]:
        """Load model from GGUF file."""
        logger.info("Loading from GGUF", path=gguf_path)

        try:
            from llama_cpp import Llama

            # Download if it's a URL
            if gguf_path.startswith(("http://", "https://")):
                import requests

                logger.info("Downloading GGUF file", url=gguf_path)
                response = requests.get(gguf_path, stream=True)
                response.raise_for_status()

                # Save to temp file
                temp_path = Path(tempfile.gettempdir()) / Path(gguf_path).name
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                gguf_path = str(temp_path)

            # Load with llama-cpp-python
            n_gpu_layers = -1 if device == "cuda" else 0
            model = Llama(
                model_path=gguf_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=kwargs.get("n_ctx", 2048),
                n_batch=kwargs.get("n_batch", 512),
                verbose=False,
            )

            # Create tokenizer wrapper
            from slm_builder.models.gguf_wrapper import GGUFModelWrapper

            wrapper = GGUFModelWrapper(model, device)
            tokenizer = wrapper.get_tokenizer()

            logger.info("GGUF model loaded", path=gguf_path)
            return wrapper, tokenizer

        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with: " "pip install llama-cpp-python"
            )
        except Exception as e:
            logger.error("Failed to load GGUF model", error=str(e))
            raise RuntimeError(f"Failed to load GGUF model from '{gguf_path}': {e}")

    @staticmethod
    def _load_from_url(
        url: str,
        device: str = "cpu",
        use_auth_token: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """Load model from HTTP/S3 URL."""
        logger.info("Loading from URL", url=url)

        # Download to temporary directory
        import requests

        temp_dir = Path(tempfile.mkdtemp())
        logger.info("Downloading model", url=url, dest=str(temp_dir))

        try:
            # Check if it's a directory or single file
            if url.endswith("/"):
                # Directory - need to download multiple files
                raise NotImplementedError(
                    "Directory URLs not yet supported. "
                    "Please provide direct file URLs or use HuggingFace Hub."
                )
            else:
                # Single file
                response = requests.get(url, stream=True)
                response.raise_for_status()

                filename = Path(url).name
                file_path = temp_dir / filename

                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Assume it's a HuggingFace model if it's a directory structure
                return ModelFactory._load_from_huggingface(
                    str(temp_dir), device, use_auth_token, trust_remote_code, **kwargs
                )

        except Exception as e:
            logger.error("Failed to load from URL", error=str(e))
            raise RuntimeError(f"Failed to load model from URL '{url}': {e}")

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
