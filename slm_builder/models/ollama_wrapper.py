"""Wrapper for Ollama models to provide HuggingFace-like interface."""

import subprocess
from typing import Dict, List

from slm_builder.utils import get_logger

logger = get_logger(__name__)


class OllamaTokenizer:
    """Simple tokenizer wrapper for Ollama models."""

    def __init__(self, model_name: str):
        """Initialize tokenizer.

        Args:
            model_name: Ollama model name
        """
        self.model_name = model_name
        self.pad_token = "<|endoftext|>"
        self.eos_token = "<|endoftext|>"
        self.bos_token = "<|startoftext|>"

    def __call__(self, text: str, **kwargs) -> Dict[str, List[int]]:
        """Tokenize text (simplified - returns dummy tokens).

        Args:
            text: Text to tokenize
            **kwargs: Additional arguments

        Returns:
            Dictionary with input_ids
        """
        # Ollama doesn't expose tokenization directly
        # Return a simplified representation
        tokens = [len(text)]  # Use length as proxy
        return {"input_ids": tokens}

    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode tokens (not fully implemented).

        Args:
            token_ids: Token IDs
            **kwargs: Additional arguments

        Returns:
            Decoded string
        """
        return ""

    def save_pretrained(self, path: str):
        """Save tokenizer (no-op for Ollama).

        Args:
            path: Output path
        """
        logger.info("Ollama tokenizer save skipped", path=path)


class OllamaModelWrapper:
    """Wrapper for Ollama models to provide training-compatible interface."""

    def __init__(self, model_name: str, device: str = "cpu"):
        """Initialize Ollama model wrapper.

        Args:
            model_name: Ollama model name
            device: Device (ignored for Ollama)
        """
        self.model_name = model_name
        self.device = device
        self._tokenizer = OllamaTokenizer(model_name)

        logger.info("Ollama model wrapper created", model=model_name)

    def get_tokenizer(self):
        """Get the tokenizer.

        Returns:
            Tokenizer instance
        """
        return self._tokenizer

    def generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> str:
        """Generate text using Ollama.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        try:
            cmd = ["ollama", "run", self.model_name, prompt]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.error("Ollama generation failed", error=result.stderr)
                return ""

        except Exception as e:
            logger.error("Ollama generation error", error=str(e))
            return ""

    def forward(self, input_ids, **kwargs):
        """Forward pass (not implemented - Ollama doesn't support this).

        Args:
            input_ids: Input token IDs
            **kwargs: Additional arguments

        Raises:
            NotImplementedError: Ollama models don't support direct training
        """
        raise NotImplementedError(
            "Ollama models cannot be trained directly. "
            "Export the model or use a HuggingFace-compatible model for training."
        )

    def save_pretrained(self, path: str, **kwargs):
        """Save model (not supported for Ollama).

        Args:
            path: Output path
            **kwargs: Additional arguments
        """
        logger.warning("Ollama models cannot be saved directly", model=self.model_name, path=path)

    def eval(self):
        """Set model to evaluation mode (no-op for Ollama).

        Returns:
            Self
        """
        return self

    def to(self, device):
        """Move model to device (no-op for Ollama).

        Args:
            device: Target device

        Returns:
            Self
        """
        return self

    def parameters(self):
        """Get model parameters (returns empty iterator).

        Returns:
            Empty iterator
        """
        return iter([])

    def __call__(self, prompt: str, **kwargs) -> str:
        """Call the model for inference.

        Args:
            prompt: Input prompt
            **kwargs: Generation parameters

        Returns:
            Generated text
        """
        return self.generate(prompt, **kwargs)
