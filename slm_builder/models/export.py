"""Model export utilities."""

from pathlib import Path
from typing import Any, Optional

from slm_builder.utils import get_logger

logger = get_logger(__name__)


def export_to_onnx(
    model_dir: str,
    output_path: str,
    opset_version: int = 13,
    optimize: bool = True,
    quantize: bool = False,
) -> str:
    """Export model to ONNX format.

    Args:
        model_dir: Directory with saved model
        output_path: Output ONNX file path
        opset_version: ONNX opset version
        optimize: Whether to optimize the model
        quantize: Whether to quantize the model

    Returns:
        Path to exported ONNX model
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("transformers and torch required for ONNX export")

    logger.info("Exporting to ONNX", model_dir=model_dir, output=output_path)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()

    # Create dummy input
    dummy_text = "Hello, this is a test."
    inputs = tokenizer(dummy_text, return_tensors="pt")

    # Export
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (inputs["input_ids"],),
                str(output_path),
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch", 1: "sequence"},
                },
                opset_version=opset_version,
                do_constant_folding=True,
            )
        logger.info("ONNX export successful", path=output_path)
    except Exception as e:
        logger.error("ONNX export failed", error=str(e))
        logger.warning("Falling back to optimum for ONNX export")

        try:
            from optimum.onnxruntime import ORTModelForCausalLM

            ort_model = ORTModelForCausalLM.from_pretrained(model_dir, export=True)
            ort_model.save_pretrained(output_path_obj.parent)
            logger.info("ONNX export via optimum successful")
        except Exception as e2:
            logger.error("Optimum export also failed", error=str(e2))
            raise

    # Quantize if requested
    if quantize:
        try:
            quantized_path = quantize_onnx_model(str(output_path))
            logger.info("Model quantized", path=quantized_path)
            return quantized_path
        except Exception as e:
            logger.warning("Quantization failed", error=str(e))

    return str(output_path)


def quantize_onnx_model(onnx_path: str, output_path: Optional[str] = None) -> str:
    """Quantize ONNX model to int8.

    Args:
        onnx_path: Path to ONNX model
        output_path: Optional output path (defaults to *_quantized.onnx)

    Returns:
        Path to quantized model
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        raise ImportError("onnxruntime required for quantization")

    if output_path is None:
        path = Path(onnx_path)
        output_path = str(path.parent / f"{path.stem}_quantized.onnx")

    logger.info("Quantizing ONNX model", input=onnx_path, output=output_path)

    quantize_dynamic(
        model_input=onnx_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
    )

    logger.info("Quantization complete", path=output_path)
    return output_path


def export_to_torchscript(
    model_dir: str,
    output_path: str,
) -> str:
    """Export model to TorchScript format.

    Args:
        model_dir: Directory with saved model
        output_path: Output TorchScript file path

    Returns:
        Path to exported TorchScript model
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("transformers and torch required for TorchScript export")

    logger.info("Exporting to TorchScript", model_dir=model_dir, output=output_path)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()

    # Create dummy input
    dummy_text = "Hello, this is a test."
    inputs = tokenizer(dummy_text, return_tensors="pt")

    # Trace model
    with torch.no_grad():
        traced_model = torch.jit.trace(model, (inputs["input_ids"],))

    # Save
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    traced_model.save(str(output_path))

    logger.info("TorchScript export successful", path=output_path)
    return str(output_path)


def create_model_bundle(
    model_dir: str,
    output_dir: str,
    format: str = "huggingface",
    optimize_for: str = "cpu",
    quantize: bool = False,
    merge_lora: bool = True,
) -> str:
    """Create a complete model bundle for deployment.

    Args:
        model_dir: Source model directory
        output_dir: Output bundle directory
        format: Export format (huggingface, onnx, torchscript)
        optimize_for: Optimization target (cpu, cuda)
        quantize: Whether to quantize
        merge_lora: Whether to merge LoRA adapters (if applicable)

    Returns:
        Path to model bundle
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Creating model bundle", format=format, optimize_for=optimize_for, quantize=quantize
    )

    if format == "huggingface":
        # Simply copy the model directory
        import shutil

        shutil.copytree(model_dir, output_dir, dirs_exist_ok=True)
        logger.info("HuggingFace bundle created", path=output_dir)
        return output_dir

    elif format == "onnx":
        onnx_path = str(output_path / "model.onnx")
        return export_to_onnx(
            model_dir,
            onnx_path,
            quantize=quantize,
        )

    elif format == "torchscript":
        ts_path = str(output_path / "model.pt")
        return export_to_torchscript(model_dir, ts_path)

    else:
        raise ValueError(f"Unsupported format: {format}")
