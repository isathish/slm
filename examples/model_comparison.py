"""Example: Compare multiple models."""

from slm_builder import SLMBuilder
from slm_builder.data import load_dataset, split_dataset
from slm_builder.models import ExperimentTracker, ModelComparator


def main():
    """Compare multiple base models on same dataset."""
    print("📊 Model Comparison Example")
    print("=" * 50)

    # Load and split dataset
    print("\n1. Loading and splitting dataset...")
    dataset = load_dataset("examples/data/sample_qa.csv", source_type="csv", task="qa")
    train, test = split_dataset(dataset, test_size=0.2, shuffle=True, random_state=42)

    print(f"   Train: {len(train)} samples")
    print(f"   Test: {len(test)} samples")

    # Train multiple models
    print("\n2. Training multiple models...")
    models_to_compare = []

    base_models = ["gpt2", "distilgpt2"]

    for base_model in base_models:
        print(f"\n   Training {base_model}...")

        builder = SLMBuilder(project_name=f"{base_model}-qa", base_model=base_model, device="auto")

        result = builder.build_from_dataset(records=train, task="qa", recipe="lora")

        print(f"   ✓ {base_model} trained: {result['model_dir']}")

        # Add to comparison list
        models_to_compare.append((base_model, builder.model, builder.tokenizer))

    # Compare models
    print("\n3. Comparing models on test set...")
    comparator = ModelComparator(output_dir="./model_comparisons")

    comparison_results = comparator.compare_models(
        models=models_to_compare,
        dataset=test,
        metrics=["perplexity", "accuracy", "bleu"],
        batch_size=8,
    )

    # Display results
    print("\n📈 Comparison Results:")
    print("-" * 50)
    for model_name, results in comparison_results["models"].items():
        if "error" in results:
            print(f"\n{model_name}: ERROR - {results['error']}")
            continue

        metrics = results["metrics"]
        print(f"\n{model_name}:")
        print(f"  Perplexity: {metrics.get('perplexity', 'N/A'):.4f}")
        print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"  BLEU: {metrics.get('bleu', 'N/A'):.4f}")
        print(f"  Evaluation time: {results['evaluation_time']:.2f}s")
        print(f"  Samples/sec: {results['samples_per_second']:.2f}")

    # Display rankings
    print("\n🏆 Rankings:")
    print("-" * 50)
    for metric, rankings in comparison_results["rankings"].items():
        print(f"\n{metric.capitalize()}:")
        for model_name, rank in sorted(rankings.items(), key=lambda x: x[1]):
            print(f"  {rank}. {model_name}")

    # Generate report
    print("\n4. Generating reports...")
    markdown_report = comparator.generate_comparison_report(comparison_results, format="markdown")

    with open("model_comparison_report.md", "w") as f:
        f.write(markdown_report)
    print("   ✓ Markdown report saved: model_comparison_report.md")

    html_report = comparator.generate_comparison_report(comparison_results, format="html")

    with open("model_comparison_report.html", "w") as f:
        f.write(html_report)
    print("   ✓ HTML report saved: model_comparison_report.html")


def experiment_tracking_example():
    """Example of experiment tracking."""
    print("\n🔬 Experiment Tracking Example")
    print("=" * 50)

    # Initialize tracker
    tracker = ExperimentTracker(tracking_dir="./experiments")

    # Load dataset
    dataset = load_dataset("examples/data/sample_qa.csv", source_type="csv", task="qa")
    train, test = split_dataset(dataset, test_size=0.2, random_state=42)

    # Try different hyperparameters
    learning_rates = [1e-5, 5e-5, 1e-4]

    for lr in learning_rates:
        print(f"\n📝 Training with LR={lr}")

        builder = SLMBuilder(project_name=f"lr_{lr}_experiment", base_model="gpt2", device="auto")

        # Build with custom hyperparameters
        result = builder.build_from_dataset(
            records=train, task="qa", recipe="lora", overrides={"training": {"learning_rate": lr}}
        )

        # Log experiment
        experiment_id = tracker.log_experiment(
            experiment_name="learning_rate_sweep",
            model_name="gpt2",
            hyperparameters={"learning_rate": lr, "batch_size": 8, "epochs": 3, "lora_r": 8},
            metrics=result["metrics"],
            notes=f"Testing learning rate {lr}",
        )

        print(f"   Logged as: {experiment_id}")

    # List all experiments
    print("\n📋 All Experiments:")
    experiments = tracker.list_experiments()
    for exp in experiments:
        print(f"  {exp['id']}: {exp['name']}")
        print(f"    LR: {exp['hyperparameters']['learning_rate']}")
        print(f"    Metrics: {exp['metrics']}")

    # Find best experiment
    best_exp = tracker.get_best_experiment(
        metric="train_loss", minimize=True  # Lower loss is better
    )

    if best_exp:
        print("\n🏅 Best Experiment:")
        print(f"  ID: {best_exp['id']}")
        print(f"  Learning Rate: {best_exp['hyperparameters']['learning_rate']}")
        print(f"  Train Loss: {best_exp['metrics']['train_loss']}")


if __name__ == "__main__":
    main()
    # Uncomment to run experiment tracking example:
    # experiment_tracking_example()
