"""Wrapper for GGUF models to provide HuggingFace-like interface."""

from typing import Any, Dict, List

from slm_builder.utils import get_logger

logger = get_logger(__name__)


class GGUFTokenizer:
    """Simple tokenizer wrapper for GGUF models."""

    def __init__(self, llama_model: Any):
        """Initialize tokenizer.

        Args:
            llama_model: llama-cpp-python Llama instance
        """
        self.llama_model = llama_model
        self.pad_token = ""
        self.eos_token = ""
        self.bos_token = ""

    def __call__(self, text: str, **kwargs) -> Dict[str, List[int]]:
        """Tokenize text using llama.cpp tokenizer.

        Args:
            text: Text to tokenize
            **kwargs: Additional arguments

        Returns:
            Dictionary with input_ids
        """
        try:
            tokens = self.llama_model.tokenize(text.encode("utf-8"))
            return {"input_ids": tokens}
        except Exception as e:
            logger.warning("Tokenization failed", error=str(e))
            return {"input_ids": []}

    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode tokens to text.

        Args:
            token_ids: Token IDs
            **kwargs: Additional arguments

        Returns:
            Decoded string
        """
        try:
            return self.llama_model.detokenize(token_ids).decode("utf-8")
        except Exception as e:
            logger.warning("Detokenization failed", error=str(e))
            return ""

    def save_pretrained(self, path: str):
        """Save tokenizer (no-op for GGUF).

        Args:
            path: Output path
        """
        logger.info("GGUF tokenizer save skipped", path=path)


class GGUFModelWrapper:
    """Wrapper for GGUF models to provide HuggingFace-like interface."""

    def __init__(self, llama_model: Any, device: str = "cpu"):
        """Initialize GGUF model wrapper.

        Args:
            llama_model: llama-cpp-python Llama instance
            device: Device
        """
        self.llama_model = llama_model
        self.device = device
        self._tokenizer = GGUFTokenizer(llama_model)

        logger.info("GGUF model wrapper created")

    def get_tokenizer(self):
        """Get the tokenizer.

        Returns:
            Tokenizer instance
        """
        return self._tokenizer

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """Generate text using GGUF model.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        try:
            output = self.llama_model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=kwargs.get("stop", []),
                echo=False,
            )

            return output["choices"][0]["text"]

        except Exception as e:
            logger.error("GGUF generation error", error=str(e))
            return ""

    def forward(self, input_ids, **kwargs):
        """Forward pass (not fully implemented for GGUF).

        Args:
            input_ids: Input token IDs
            **kwargs: Additional arguments

        Raises:
            NotImplementedError: GGUF models have limited training support
        """
        raise NotImplementedError(
            "GGUF models don't support direct training via standard methods. "
            "Consider converting to a HuggingFace-compatible format."
        )

    def save_pretrained(self, path: str, **kwargs):
        """Save model (limited support for GGUF).

        Args:
            path: Output path
            **kwargs: Additional arguments
        """
        logger.warning("GGUF models cannot be saved in HuggingFace format", path=path)

    def eval(self):
        """Set model to evaluation mode (no-op for GGUF).

        Returns:
            Self
        """
        return self

    def to(self, device):
        """Move model to device (no-op for GGUF - handled at init).

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
