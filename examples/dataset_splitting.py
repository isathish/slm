"""Example: Dataset splitting and validation."""

from slm_builder import SLMBuilder
from slm_builder.data import load_dataset, split_dataset, validate_dataset


def main():
    """Example of dataset splitting and validation."""
    print("🔀 Dataset Splitting and Validation")
    print("=" * 50)

    # Load dataset
    print("\n1. Loading dataset...")
    dataset = load_dataset("examples/data/sample_qa.csv", source_type="csv", task="qa")
    print(f"   Loaded {len(dataset)} records")

    # Validate dataset quality
    print("\n2. Validating dataset...")
    report = validate_dataset(dataset, task="qa", strict=False)

    print(f"   Valid samples: {report['valid_samples']}/{report['total_samples']}")
    print(f"   Validity rate: {report['validity_rate']:.2%}")
    print(f"   Errors: {len(report['errors'])}")
    print(f"   Warnings: {len(report['warnings'])}")

    if report["statistics"]:
        stats = report["statistics"]
        print(f"   Mean text length: {stats['mean_text_length']:.1f} chars")
        print(f"   Min/Max length: {stats['min_text_length']}/{stats['max_text_length']}")

    # Split dataset with stratification
    print("\n3. Splitting dataset (train/val/test)...")
    train, val, test = split_dataset(
        dataset,
        test_size=0.15,
        val_size=0.15,
        stratify_by="label.label",  # Maintain class distribution
        shuffle=True,
        random_state=42,
    )

    print(f"   Train: {len(train)} samples")
    print(f"   Val: {len(val)} samples")
    print(f"   Test: {len(test)} samples")

    # Train model on split data
    print("\n4. Training model on split data...")
    builder = SLMBuilder(project_name="split-model", base_model="gpt2", device="auto")

    result = builder.build_from_dataset(records=train, task="qa", recipe="lora")

    print("\n✅ Training complete!")
    print(f"Model saved to: {result['model_dir']}")

    # Evaluate on validation set
    print("\n5. Evaluating on validation set...")
    from slm_builder.models import evaluate_model

    model = builder.model
    tokenizer = builder.tokenizer

    eval_results = evaluate_model(
        model, tokenizer, val, metrics=["perplexity", "accuracy"], batch_size=8
    )

    print(f"   Validation metrics: {eval_results}")


def k_fold_example():
    """Example of K-fold cross-validation."""
    from slm_builder.data import DatasetSplitter

    print("\n🔄 K-Fold Cross-Validation Example")
    print("=" * 50)

    # Load dataset
    dataset = load_dataset("examples/data/sample_qa.csv", source_type="csv", task="qa")

    # Create 5-fold splits
    splitter = DatasetSplitter()
    folds = splitter.k_fold_split(dataset, n_folds=5, shuffle=True, random_state=42)

    print(f"Created {len(folds)} folds")

    # Train on each fold
    for fold_idx, (train_fold, val_fold) in enumerate(folds):
        print(f"\nFold {fold_idx + 1}:")
        print(f"  Train: {len(train_fold)} samples")
        print(f"  Val: {len(val_fold)} samples")

        builder = SLMBuilder(
            project_name=f"fold{fold_idx+1}-model", base_model="distilgpt2", device="auto"
        )

        result = builder.build_from_dataset(records=train_fold, task="qa", recipe="lora")

        print(f"  Model saved: {result['model_dir']}")
        print(f"  Metrics: {result['metrics']}")


def class_balance_example():
    """Example of checking class balance."""
    from slm_builder.data import DatasetValidator

    print("\n⚖️ Class Balance Check Example")
    print("=" * 50)

    # Load classification dataset
    dataset = load_dataset(
        "examples/data/sample_classification.csv", source_type="csv", task="classification"
    )

    # Check class distribution
    validator = DatasetValidator()
    distribution = validator.check_class_balance(dataset, stratify_by="label.label")

    print(f"Total samples: {distribution['total_samples']}")
    print(f"Number of classes: {distribution['n_classes']}")
    print(f"Balanced: {distribution['is_balanced']}")
    print(f"Imbalance ratio: {distribution['imbalance_ratio']:.2f}")

    print("\nClass distribution:")
    for class_name, count in distribution["class_counts"].items():
        percentage = distribution["class_percentages"][class_name]
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")


if __name__ == "__main__":
    main()
    # Uncomment to run other examples:
    # k_fold_example()
    # class_balance_example()
