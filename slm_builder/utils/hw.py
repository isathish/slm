"""Hardware detection and resource recommendations."""

import platform
from typing import Dict, List, Optional

import psutil


def detect_hardware() -> Dict[str, any]:
    """Detect available hardware resources.
    
    Returns:
        Dictionary with hardware information:
        - has_cuda: bool indicating CUDA availability
        - gpu_count: number of GPUs
        - gpu_memory_gb: total GPU memory in GB (first GPU)
        - cpu_count: number of physical CPU cores
        - ram_gb: total RAM in GB
        - platform: OS platform
    """
    hw_info = {
        "has_cuda": False,
        "gpu_count": 0,
        "gpu_memory_gb": 0.0,
        "cpu_count": psutil.cpu_count(logical=False) or 1,
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "platform": platform.system(),
    }

    try:
        import torch

        hw_info["has_cuda"] = torch.cuda.is_available()
        hw_info["gpu_count"] = torch.cuda.device_count() if hw_info["has_cuda"] else 0

        if hw_info["has_cuda"]:
            props = torch.cuda.get_device_properties(0)
            hw_info["gpu_memory_gb"] = round(props.total_memory / (1024**3), 2)
            hw_info["gpu_name"] = props.name
    except ImportError:
        pass  # torch not installed, CPU-only mode

    return hw_info


def recommend_base_models(hw_profile: Dict[str, any]) -> List[Dict[str, str]]:
    """Recommend base models based on hardware profile.
    
    Args:
        hw_profile: Hardware profile from detect_hardware()
        
    Returns:
        List of recommended models with name, size, and reason
    """
    recommendations = []

    if hw_profile["has_cuda"]:
        gpu_mem = hw_profile["gpu_memory_gb"]
        if gpu_mem >= 24:
            recommendations.extend([
                {
                    "name": "meta-llama/Llama-2-7b-hf",
                    "size": "7B",
                    "reason": "Good balance of capability and speed with 24GB+ GPU",
                },
                {
                    "name": "mistralai/Mistral-7B-v0.1",
                    "size": "7B",
                    "reason": "High performance 7B model for large GPU",
                },
            ])
        elif gpu_mem >= 16:
            recommendations.extend([
                {
                    "name": "microsoft/phi-2",
                    "size": "2.7B",
                    "reason": "Efficient 2.7B model suitable for 16GB GPU",
                },
                {
                    "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "size": "1.1B",
                    "reason": "Compact model with good performance for 16GB GPU",
                },
            ])
        else:  # 8-12GB
            recommendations.extend([
                {
                    "name": "gpt2",
                    "size": "124M",
                    "reason": "Lightweight, well-supported baseline for smaller GPUs",
                },
                {
                    "name": "gpt2-medium",
                    "size": "355M",
                    "reason": "Good balance for GPUs with 8-12GB",
                },
            ])
    else:
        # CPU-only recommendations
        ram_gb = hw_profile["ram_gb"]
        if ram_gb >= 16:
            recommendations.extend([
                {
                    "name": "gpt2-medium",
                    "size": "355M",
                    "reason": "Reasonable CPU inference with 16GB+ RAM",
                },
                {
                    "name": "distilgpt2",
                    "size": "82M",
                    "reason": "Fast CPU inference, good for production",
                },
            ])
        else:
            recommendations.extend([
                {
                    "name": "gpt2",
                    "size": "124M",
                    "reason": "Best for CPU-only with limited RAM",
                },
                {
                    "name": "distilgpt2",
                    "size": "82M",
                    "reason": "Fastest CPU inference option",
                },
            ])

    return recommendations


def recommend_recipe(
    hw_profile: Dict[str, any], user_constraints: Optional[Dict[str, any]] = None
) -> str:
    """Recommend training recipe based on hardware and constraints.
    
    Args:
        hw_profile: Hardware profile from detect_hardware()
        user_constraints: Optional user constraints (time_budget, quality_target, etc.)
        
    Returns:
        Recommended recipe name
    """
    user_constraints = user_constraints or {}

    # If user explicitly wants full finetune and has GPU, allow it
    if user_constraints.get("prefer_finetune") and hw_profile["has_cuda"]:
        return "finetune"

    # If CPU-only or limited GPU, recommend LoRA
    if not hw_profile["has_cuda"]:
        return "lora"

    gpu_mem = hw_profile["gpu_memory_gb"]

    # Small GPU (< 12GB): LoRA only
    if gpu_mem < 12:
        return "lora"

    # Medium GPU (12-20GB): LoRA is safer, but finetune possible for small models
    if gpu_mem < 20:
        return "lora"

    # Large GPU (20GB+): Can do full finetune
    if user_constraints.get("max_quality"):
        return "finetune"

    # Default to LoRA for efficiency
    return "lora"


def recommend_batch_size(hw_profile: Dict[str, any], recipe: str) -> int:
    """Recommend batch size based on hardware and recipe.
    
    Args:
        hw_profile: Hardware profile from detect_hardware()
        recipe: Training recipe name
        
    Returns:
        Recommended batch size
    """
    if not hw_profile["has_cuda"]:
        # CPU-only: small batch sizes
        return 4 if recipe == "finetune" else 8

    gpu_mem = hw_profile["gpu_memory_gb"]

    if recipe == "lora":
        if gpu_mem >= 24:
            return 16
        elif gpu_mem >= 16:
            return 12
        elif gpu_mem >= 12:
            return 8
        else:
            return 4
    else:  # finetune
        if gpu_mem >= 24:
            return 8
        elif gpu_mem >= 16:
            return 4
        else:
            return 2

    return 8  # fallback


def estimate_training_time(
    dataset_size: int,
    hw_profile: Dict[str, any],
    batch_size: int,
    epochs: int,
    recipe: str,
) -> str:
    """Estimate training time based on dataset and hardware.
    
    Args:
        dataset_size: Number of training examples
        hw_profile: Hardware profile
        batch_size: Training batch size
        epochs: Number of epochs
        recipe: Training recipe
        
    Returns:
        Human-readable time estimate
    """
    # Very rough estimates based on typical throughput
    steps = (dataset_size // batch_size) * epochs

    if hw_profile["has_cuda"]:
        gpu_mem = hw_profile["gpu_memory_gb"]
        if gpu_mem >= 24:
            seconds_per_step = 0.5 if recipe == "lora" else 1.5
        elif gpu_mem >= 16:
            seconds_per_step = 1.0 if recipe == "lora" else 3.0
        else:
            seconds_per_step = 2.0 if recipe == "lora" else 5.0
    else:
        # CPU is much slower
        seconds_per_step = 10.0 if recipe == "lora" else 30.0

    total_seconds = int(steps * seconds_per_step)

    if total_seconds < 60:
        return f"~{total_seconds} seconds"
    elif total_seconds < 3600:
        return f"~{total_seconds // 60} minutes"
    else:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"~{hours}h {minutes}m"


def get_device_string(hw_profile: Optional[Dict[str, any]] = None, device: Optional[str] = None) -> str:
    """Get appropriate device string for PyTorch.
    
    Args:
        hw_profile: Optional hardware profile
        device: Optional explicit device preference ('cpu', 'cuda', 'auto')
        
    Returns:
        Device string ('cpu' or 'cuda')
    """
    if device == "cpu":
        return "cpu"

    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    # Auto-detect
    if hw_profile is None:
        hw_profile = detect_hardware()

    return "cuda" if hw_profile["has_cuda"] else "cpu"
