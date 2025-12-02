"""Configuration management and validation for SLM Builder."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class PreprocessConfig(BaseModel):
    """Preprocessing configuration."""

    max_tokens_per_chunk: int = Field(default=512, ge=1, le=4096)
    chunk_overlap: int = Field(default=64, ge=0)
    lowercase: bool = False
    strip_urls: bool = True
    remove_duplicates: bool = True
    normalize_unicode: bool = True


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    batch_size: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=5e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    epochs: int = Field(default=3, ge=1)
    warmup_steps: int = Field(default=100, ge=0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    seed: int = Field(default=42, ge=0)
    fp16: bool = False
    eval_steps: int = Field(default=500, ge=1)
    save_steps: int = Field(default=500, ge=1)
    logging_steps: int = Field(default=100, ge=1)
    save_total_limit: int = Field(default=3, ge=1)


class LoRAConfig(BaseModel):
    """LoRA-specific configuration."""

    r: int = Field(default=8, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.1, ge=0, le=1)
    target_modules: Optional[List[str]] = None
    bias: str = Field(default="none")
    task_type: str = Field(default="CAUSAL_LM")

    @field_validator("target_modules", mode="before")
    @classmethod
    def set_default_target_modules(cls, v):
        if v is None:
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        return v


class ExportConfig(BaseModel):
    """Export configuration."""

    format: str = Field(default="onnx")
    optimize_for: str = Field(default="cpu")
    quantize: bool = False
    opset_version: int = Field(default=13, ge=11)
    merge_lora: bool = True

    @field_validator("format")
    @classmethod
    def validate_format(cls, v):
        allowed = ["onnx", "torchscript", "huggingface"]
        if v not in allowed:
            raise ValueError(f"format must be one of {allowed}")
        return v

    @field_validator("optimize_for")
    @classmethod
    def validate_optimize_for(cls, v):
        allowed = ["cpu", "cuda", "auto"]
        if v not in allowed:
            raise ValueError(f"optimize_for must be one of {allowed}")
        return v


class SLMConfig(BaseModel):
    """Main SLM Builder configuration."""

    project_name: str
    base_model: str = Field(default="gpt2")
    task: str = Field(default="qa")
    recipe: str = Field(default="lora")
    device: Optional[str] = None
    work_dir: str = Field(default="./slm_workdir")
    
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    # Security and governance
    allow_pii: bool = False
    check_license: bool = True
    
    # Additional options
    use_auth_token: Optional[str] = None
    trust_remote_code: bool = False

    @field_validator("task")
    @classmethod
    def validate_task(cls, v):
        allowed = ["qa", "classification", "generation", "instruction"]
        if v not in allowed:
            raise ValueError(f"task must be one of {allowed}")
        return v

    @field_validator("recipe")
    @classmethod
    def validate_recipe(cls, v):
        allowed = ["lora", "finetune", "instruction-tune", "quantize", "distill"]
        if v not in allowed:
            raise ValueError(f"recipe must be one of {allowed}")
        return v

    @field_validator("device", mode="before")
    @classmethod
    def validate_device(cls, v):
        if v is not None and v not in ["cpu", "cuda", "auto"]:
            raise ValueError("device must be one of ['cpu', 'cuda', 'auto', None]")
        return v


# Default configurations for different recipes
DEFAULT_RECIPE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "lora": {
        "training": {"epochs": 3, "learning_rate": 5e-5, "batch_size": 8},
        "lora": {"r": 8, "lora_alpha": 32, "lora_dropout": 0.1},
    },
    "finetune": {
        "training": {"epochs": 5, "learning_rate": 2e-5, "batch_size": 16},
    },
    "instruction-tune": {
        "training": {"epochs": 3, "learning_rate": 3e-5, "batch_size": 8},
    },
    "quantize": {
        "export": {"quantize": True, "format": "onnx"},
    },
}


def get_recipe_defaults(recipe: str) -> Dict[str, Any]:
    """Get default configuration overrides for a specific recipe.
    
    Args:
        recipe: Recipe name (lora, finetune, etc.)
        
    Returns:
        Dictionary with recipe-specific config overrides
    """
    return DEFAULT_RECIPE_CONFIGS.get(recipe, {})


def merge_configs(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge configuration dictionaries.
    
    Args:
        base: Base configuration dictionary
        overrides: Configuration overrides
        
    Returns:
        Merged configuration dictionary
    """
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result
