"""Example: Build a QA system from CSV data."""

from slm_builder import SLMBuilder


def main():
    # Initialize builder
    print("🚀 Initializing SLM Builder...")
    builder = SLMBuilder(
        project_name="faq-bot",
        base_model="gpt2",
        device="auto",  # Auto-detect CPU/GPU
    )

    # Build from CSV
    print("📚 Building SLM from CSV...")
    result = builder.build_from_csv(
        path="data/faqs.csv",  # CSV with 'question' and 'answer' columns
        task="qa",
        recipe="lora",
        overrides={
            "training": {
                "epochs": 3,
                "batch_size": 8,
            }
        }
    )

    print("\n✅ Training complete!")
    print(f"   Model: {result['model_dir']}")
    print(f"   Metrics: {result['metrics']}")

    # Export to ONNX for CPU deployment
    print("\n📦 Exporting to ONNX...")
    exported = builder.export(
        model_dir=result['model_dir'],
        format="onnx",
        optimize_for="cpu",
    )
    print(f"   Exported: {exported}")

    # Optionally serve the model
    print("\n🌐 To serve the model, run:")
    print(f"   slm serve --model {result['model_dir']} --port 8080")


if __name__ == "__main__":
    main()
