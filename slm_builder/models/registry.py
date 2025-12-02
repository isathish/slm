"""Model registry for discovering and loading models from various sources."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from slm_builder.utils import get_logger

logger = get_logger(__name__)


class ModelSource:
    """Enum-like class for model sources."""

    HUGGINGFACE_HUB = "huggingface_hub"
    LOCAL_PATH = "local_path"
    OLLAMA = "ollama"
    GGUF_FILE = "gguf_file"
    HTTP_URL = "http_url"
    S3_URL = "s3_url"
    MODEL_ZOO = "model_zoo"
    CUSTOM = "custom"


class ModelRegistry:
    """Registry for managing model sources and discovery."""

    # Popular model collections
    MODEL_ZOO = {
        # GPT family
        "gpt2": {"source": ModelSource.HUGGINGFACE_HUB, "path": "gpt2", "size": "124M"},
        "gpt2-medium": {
            "source": ModelSource.HUGGINGFACE_HUB,
            "path": "gpt2-medium",
            "size": "355M",
        },
        "gpt2-large": {
            "source": ModelSource.HUGGINGFACE_HUB,
            "path": "gpt2-large",
            "size": "774M",
        },
        "distilgpt2": {
            "source": ModelSource.HUGGINGFACE_HUB,
            "path": "distilgpt2",
            "size": "82M",
        },
        # Llama family
        "llama-2-7b": {
            "source": ModelSource.HUGGINGFACE_HUB,
            "path": "meta-llama/Llama-2-7b-hf",
            "size": "7B",
        },
        "llama-2-13b": {
            "source": ModelSource.HUGGINGFACE_HUB,
            "path": "meta-llama/Llama-2-13b-hf",
            "size": "13B",
        },
        "llama-3-8b": {
            "source": ModelSource.HUGGINGFACE_HUB,
            "path": "meta-llama/Meta-Llama-3-8B",
            "size": "8B",
        },
        # Mistral family
        "mistral-7b": {
            "source": ModelSource.HUGGINGFACE_HUB,
            "path": "mistralai/Mistral-7B-v0.1",
            "size": "7B",
        },
        "mixtral-8x7b": {
            "source": ModelSource.HUGGINGFACE_HUB,
            "path": "mistralai/Mixtral-8x7B-v0.1",
            "size": "47B",
        },
        # Phi family
        "phi-2": {
            "source": ModelSource.HUGGINGFACE_HUB,
            "path": "microsoft/phi-2",
            "size": "2.7B",
        },
        "phi-3-mini": {
            "source": ModelSource.HUGGINGFACE_HUB,
            "path": "microsoft/Phi-3-mini-4k-instruct",
            "size": "3.8B",
        },
        # Gemma family
        "gemma-2b": {
            "source": ModelSource.HUGGINGFACE_HUB,
            "path": "google/gemma-2b",
            "size": "2B",
        },
        "gemma-7b": {
            "source": ModelSource.HUGGINGFACE_HUB,
            "path": "google/gemma-7b",
            "size": "7B",
        },
        # TinyLlama
        "tinyllama": {
            "source": ModelSource.HUGGINGFACE_HUB,
            "path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "size": "1.1B",
        },
        # Qwen family
        "qwen-1.8b": {
            "source": ModelSource.HUGGINGFACE_HUB,
            "path": "Qwen/Qwen-1_8B",
            "size": "1.8B",
        },
        "qwen-7b": {"source": ModelSource.HUGGINGFACE_HUB, "path": "Qwen/Qwen-7B", "size": "7B"},
    }

    def __init__(self):
        """Initialize model registry."""
        self.custom_models: Dict[str, Dict[str, Any]] = {}

    def register_custom_model(self, name: str, source: str, path: str, **metadata) -> None:
        """Register a custom model.

        Args:
            name: Model name/alias
            source: Model source type
            path: Path/URL to model
            **metadata: Additional metadata (size, description, etc.)
        """
        self.custom_models[name] = {
            "source": source,
            "path": path,
            **metadata,
        }
        logger.info("Registered custom model", name=name, source=source)

    def detect_source(self, model_identifier: str) -> Tuple[str, str]:
        """Detect model source from identifier.

        Args:
            model_identifier: Model name, path, or URL

        Returns:
            Tuple of (source_type, resolved_path)
        """
        # Check if in model zoo
        if model_identifier in self.MODEL_ZOO:
            model_info = self.MODEL_ZOO[model_identifier]
            logger.info(
                "Found model in zoo",
                name=model_identifier,
                size=model_info.get("size"),
            )
            return model_info["source"], model_info["path"]

        # Check custom models
        if model_identifier in self.custom_models:
            model_info = self.custom_models[model_identifier]
            logger.info("Found custom model", name=model_identifier)
            return model_info["source"], model_info["path"]

        # Check if it's a local path
        path = Path(model_identifier)
        if path.exists():
            if path.is_dir():
                # Check if it's a HuggingFace model directory
                if (path / "config.json").exists():
                    logger.info("Detected local HuggingFace model", path=str(path))
                    return ModelSource.LOCAL_PATH, str(path)
                else:
                    logger.info("Detected local directory", path=str(path))
                    return ModelSource.LOCAL_PATH, str(path)
            elif path.suffix == ".gguf":
                logger.info("Detected GGUF file", path=str(path))
                return ModelSource.GGUF_FILE, str(path)
            else:
                logger.info("Detected local file", path=str(path))
                return ModelSource.LOCAL_PATH, str(path)

        # Check if it's a URL
        parsed = urlparse(model_identifier)
        if parsed.scheme in ["http", "https"]:
            if model_identifier.endswith(".gguf"):
                logger.info("Detected GGUF URL", url=model_identifier)
                return ModelSource.GGUF_FILE, model_identifier
            logger.info("Detected HTTP URL", url=model_identifier)
            return ModelSource.HTTP_URL, model_identifier
        elif parsed.scheme == "s3":
            logger.info("Detected S3 URL", url=model_identifier)
            return ModelSource.S3_URL, model_identifier

        # Check if it's an Ollama model (format: ollama:model_name)
        if model_identifier.startswith("ollama:"):
            model_name = model_identifier.split(":", 1)[1]
            logger.info("Detected Ollama model", name=model_name)
            return ModelSource.OLLAMA, model_name

        # Check if it looks like HuggingFace repo (org/model format)
        if "/" in model_identifier and not model_identifier.startswith("/"):
            logger.info("Assumed HuggingFace Hub model", repo=model_identifier)
            return ModelSource.HUGGINGFACE_HUB, model_identifier

        # Default to HuggingFace Hub for simple names
        logger.info("Defaulting to HuggingFace Hub", model=model_identifier)
        return ModelSource.HUGGINGFACE_HUB, model_identifier

    def get_model_info(self, model_identifier: str) -> Dict[str, Any]:
        """Get information about a model.

        Args:
            model_identifier: Model name or identifier

        Returns:
            Dictionary with model information
        """
        source, path = self.detect_source(model_identifier)

        info = {
            "identifier": model_identifier,
            "source": source,
            "path": path,
        }

        # Add zoo metadata if available
        if model_identifier in self.MODEL_ZOO:
            info.update(self.MODEL_ZOO[model_identifier])
        elif model_identifier in self.custom_models:
            info.update(self.custom_models[model_identifier])

        return info

    def list_available_models(self, source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all available models.

        Args:
            source_filter: Optional filter by source type

        Returns:
            List of model information dictionaries
        """
        models = []

        # Add zoo models
        for name, info in self.MODEL_ZOO.items():
            if source_filter is None or info["source"] == source_filter:
                models.append({"name": name, **info})

        # Add custom models
        for name, info in self.custom_models.items():
            if source_filter is None or info["source"] == source_filter:
                models.append({"name": name, **info})

        return models

    def search_models(
        self,
        query: str,
        source: Optional[str] = None,
        size_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for models by query.

        Args:
            query: Search query (matches name, path, description)
            source: Optional source filter
            size_filter: Optional size filter (e.g., "7B", "small")

        Returns:
            List of matching models
        """
        query_lower = query.lower()
        results = []

        for model in self.list_available_models(source_filter=source):
            # Check query match
            name_match = query_lower in model["name"].lower()
            path_match = query_lower in model.get("path", "").lower()
            desc_match = query_lower in model.get("description", "").lower()

            if name_match or path_match or desc_match:
                # Apply size filter
                if size_filter:
                    model_size = model.get("size", "").lower()
                    if size_filter.lower() not in model_size:
                        continue

                results.append(model)

        return results

    @staticmethod
    def validate_huggingface_model(model_path: str, token: Optional[str] = None) -> bool:
        """Validate if a HuggingFace model exists and is accessible.

        Args:
            model_path: HuggingFace model repo path
            token: Optional authentication token

        Returns:
            True if model is valid and accessible
        """
        try:
            from huggingface_hub import model_info

            model_info(model_path, token=token)
            return True
        except Exception as e:
            logger.warning("HuggingFace model validation failed", model=model_path, error=str(e))
            return False

    @staticmethod
    def check_ollama_availability() -> bool:
        """Check if Ollama is available on the system.

        Returns:
            True if Ollama is available
        """
        import subprocess

        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def list_ollama_models() -> List[str]:
        """List available Ollama models.

        Returns:
            List of model names
        """
        import subprocess

        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # Parse output
                lines = result.stdout.strip().split("\n")[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
        except Exception as e:
            logger.warning("Failed to list Ollama models", error=str(e))

        return []


# Global registry instance
_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Get the global model registry instance.

    Returns:
        ModelRegistry instance
    """
    return _registry


def register_model(name: str, source: str, path: str, **metadata) -> None:
    """Register a model in the global registry.

    Args:
        name: Model name/alias
        source: Model source type
        path: Path/URL to model
        **metadata: Additional metadata
    """
    _registry.register_custom_model(name, source, path, **metadata)


def detect_model_source(model_identifier: str) -> Tuple[str, str]:
    """Detect model source from identifier.

    Args:
        model_identifier: Model name, path, or URL

    Returns:
        Tuple of (source_type, resolved_path)
    """
    return _registry.detect_source(model_identifier)


def list_models(source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """List available models.

    Args:
        source_filter: Optional source type filter

    Returns:
        List of model information
    """
    return _registry.list_available_models(source_filter)


def search_models(query: str, **filters) -> List[Dict[str, Any]]:
    """Search for models.

    Args:
        query: Search query
        **filters: Additional filters (source, size_filter)

    Returns:
        List of matching models
    """
    return _registry.search_models(query, **filters)
