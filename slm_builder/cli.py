"""Command-line interface for SLM Builder."""

import sys
from pathlib import Path
from typing import Optional

import click

from slm_builder import SLMBuilder
from slm_builder.data.annotator import annotate_dataset
from slm_builder.utils import load_jsonl, load_yaml, setup_logging


@click.group()
@click.version_option()
def cli():
    """SLM Builder - Build Small/Specialized Language Models from any data source."""
    setup_logging()


@cli.command()
@click.option("--source", required=True, help="Path to data source (CSV, JSONL, directory)")
@click.option(
    "--task", default="qa", help="Task type (qa, classification, generation, instruction)"
)
@click.option("--recipe", default="lora", help="Training recipe (lora, finetune, instruction-tune)")
@click.option("--base-model", default="gpt2", help="Base model name or path")
@click.option("--out", "--output", "output_dir", help="Output directory")
@click.option("--config", "config_file", help="Path to config YAML file")
@click.option("--device", help="Device (cpu, cuda, auto)")
@click.option("--project-name", default="slm-project", help="Project name")
@click.option("--work-dir", default="./slm_workdir", help="Working directory")
def build(
    source: str,
    task: str,
    recipe: str,
    base_model: str,
    output_dir: Optional[str],
    config_file: Optional[str],
    device: Optional[str],
    project_name: str,
    work_dir: str,
):
    """Build an SLM from a data source."""
    click.echo(f"🚀 Building SLM: {project_name}")
    click.echo(f"   Source: {source}")
    click.echo(f"   Task: {task}")
    click.echo(f"   Recipe: {recipe}")
    click.echo(f"   Base Model: {base_model}")

    # Load config if provided
    config = {}
    if config_file:
        config = load_yaml(config_file)
        click.echo(f"   Config loaded from: {config_file}")

    # Initialize builder
    builder = SLMBuilder(
        project_name=project_name,
        base_model=base_model,
        device=device,
        work_dir=work_dir,
        config=config,
    )

    # Determine source type and build
    source_path = Path(source)

    try:
        if source.endswith(".csv"):
            result = builder.build_from_csv(
                path=source,
                task=task,
                recipe=recipe,
                output_dir=output_dir,
            )
        elif source.endswith(".jsonl") or source.endswith(".json"):
            result = builder.build_from_jsonl(
                path=source,
                task=task,
                recipe=recipe,
                output_dir=output_dir,
            )
        elif source_path.is_dir():
            result = builder.build_from_text_dir(
                path=source,
                task=task,
                recipe=recipe,
                output_dir=output_dir,
            )
        else:
            click.echo(f"❌ Unsupported source format: {source}", err=True)
            sys.exit(1)

        click.echo("\n✅ Build complete!")
        click.echo(f"   Model directory: {result['model_dir']}")
        click.echo(f"   Output directory: {result['output_dir']}")

    except Exception as e:
        click.echo(f"\n❌ Build failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--source", required=True, help="Path to data source")
@click.option("--task", default="qa", help="Task type")
@click.option(
    "--out", "--output", "output_path", default="annotated.jsonl", help="Output JSONL file"
)
@click.option("--port", default=8501, help="Streamlit server port")
def annotate(source: str, task: str, output_path: str, port: int):
    """Launch annotation UI for labeling data."""
    click.echo("🏷️  Launching annotation UI")
    click.echo(f"   Source: {source}")
    click.echo(f"   Task: {task}")
    click.echo(f"   Output: {output_path}")

    # Load data
    from slm_builder.data import load_dataset

    try:
        records = load_dataset(source, task=task)
        click.echo(f"   Loaded {len(records)} records")

        annotate_dataset(
            records=records,
            task=task,
            output_path=output_path,
            port=port,
        )
    except Exception as e:
        click.echo(f"❌ Annotation failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--model", "model_dir", required=True, help="Path to model directory")
@click.option("--format", default="onnx", help="Export format (onnx, torchscript, huggingface)")
@click.option("--optimize", default="cpu", help="Optimize for (cpu, cuda)")
@click.option("--out", "--output", "output_dir", help="Output directory")
@click.option("--quantize", is_flag=True, help="Quantize the model")
def export(model_dir: str, format: str, optimize: str, output_dir: Optional[str], quantize: bool):
    """Export a trained model to ONNX or other formats."""
    click.echo("📦 Exporting model")
    click.echo(f"   Model: {model_dir}")
    click.echo(f"   Format: {format}")
    click.echo(f"   Optimize for: {optimize}")
    click.echo(f"   Quantize: {quantize}")

    from slm_builder.models.export import create_model_bundle

    try:
        if output_dir is None:
            output_dir = f"{model_dir}_exported"

        exported_path = create_model_bundle(
            model_dir=model_dir,
            output_dir=output_dir,
            format=format,
            optimize_for=optimize,
            quantize=quantize,
        )

        click.echo("\n✅ Export complete!")
        click.echo(f"   Exported to: {exported_path}")

    except Exception as e:
        click.echo(f"❌ Export failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--model", "model_dir", required=True, help="Path to model directory")
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=8080, help="Server port")
def serve(model_dir: str, host: str, port: int):
    """Start a FastAPI server to serve the model."""
    click.echo("🚀 Starting server")
    click.echo(f"   Model: {model_dir}")
    click.echo(f"   Host: {host}")
    click.echo(f"   Port: {port}")

    try:
        # Import and start server
        from slm_builder.serve.fastapi_server import start_server

        start_server(model_dir, host=host, port=port)

    except KeyboardInterrupt:
        click.echo("\n👋 Server stopped")
    except Exception as e:
        click.echo(f"❌ Server failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--source", required=True, help="Path to data source")
def info(source: str):
    """Display information about a dataset or model."""
    click.echo(f"ℹ️  Information for: {source}")

    source_path = Path(source)

    if source_path.is_file():
        # Dataset info
        if source.endswith(".jsonl") or source.endswith(".json"):
            records = load_jsonl(source)
            click.echo("   Type: JSONL Dataset")
            click.echo(f"   Records: {len(records)}")
            if records:
                click.echo(f"   Sample keys: {list(records[0].keys())}")
                tasks = set(r.get("task") for r in records)
                click.echo(f"   Tasks: {tasks}")
        else:
            click.echo(f"   File size: {source_path.stat().st_size / 1024:.2f} KB")

    elif source_path.is_dir():
        # Check if it's a model directory
        if (source_path / "config.json").exists():
            click.echo("   Type: HuggingFace Model")
            if (source_path / "pytorch_model.bin").exists():
                size = (source_path / "pytorch_model.bin").stat().st_size / (1024**2)
                click.echo(f"   Model size: {size:.2f} MB")
        else:
            # Directory of files
            files = list(source_path.glob("*"))
            click.echo("   Type: Directory")
            click.echo(f"   Files: {len(files)}")


if __name__ == "__main__":
    cli()
