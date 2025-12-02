"""Dataset splitting and validation utilities."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from slm_builder.utils import get_logger

logger = get_logger(__name__)


class DatasetSplitter:
    """Dataset splitting with stratification support."""

    @staticmethod
    def train_test_split(
        dataset: List[Dict[str, Any]],
        test_size: float = 0.2,
        stratify_by: Optional[str] = None,
        shuffle: bool = True,
        random_state: Optional[int] = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split dataset into train and test sets.

        Args:
            dataset: Dataset to split
            test_size: Proportion of test set (0.0 to 1.0)
            stratify_by: Field to stratify by (e.g., 'label.label' for classification)
            shuffle: Whether to shuffle before splitting
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0.0 and 1.0")

        n_samples = len(dataset)
        n_test = int(n_samples * test_size)

        logger.info(
            "Splitting dataset",
            total=n_samples,
            train=n_samples - n_test,
            test=n_test,
            stratify=stratify_by is not None,
        )

        if stratify_by:
            return DatasetSplitter._stratified_split(dataset, test_size, stratify_by, random_state)
        else:
            return DatasetSplitter._random_split(dataset, test_size, shuffle, random_state)

    @staticmethod
    def train_val_test_split(
        dataset: List[Dict[str, Any]],
        val_size: float = 0.15,
        test_size: float = 0.15,
        stratify_by: Optional[str] = None,
        shuffle: bool = True,
        random_state: Optional[int] = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split dataset into train, validation, and test sets.

        Args:
            dataset: Dataset to split
            val_size: Proportion of validation set
            test_size: Proportion of test set
            stratify_by: Field to stratify by
            shuffle: Whether to shuffle
            random_state: Random seed

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if not (0.0 < val_size < 1.0 and 0.0 < test_size < 1.0):
            raise ValueError("val_size and test_size must be between 0.0 and 1.0")

        if val_size + test_size >= 1.0:
            raise ValueError("val_size + test_size must be less than 1.0")

        n_samples = len(dataset)
        n_test = int(n_samples * test_size)
        n_val = int(n_samples * val_size)
        n_train = n_samples - n_val - n_test

        logger.info(
            "Splitting dataset into train/val/test",
            total=n_samples,
            train=n_train,
            val=n_val,
            test=n_test,
        )

        # First split: separate test set
        train_val, test = DatasetSplitter.train_test_split(
            dataset,
            test_size=test_size,
            stratify_by=stratify_by,
            shuffle=shuffle,
            random_state=random_state,
        )

        # Second split: separate validation from training
        val_size_adjusted = val_size / (1.0 - test_size)
        train, val = DatasetSplitter.train_test_split(
            train_val,
            test_size=val_size_adjusted,
            stratify_by=stratify_by,
            shuffle=shuffle,
            random_state=random_state + 1 if random_state else None,
        )

        return train, val, test

    @staticmethod
    def k_fold_split(
        dataset: List[Dict[str, Any]],
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = 42,
    ) -> List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
        """Create K-fold cross-validation splits.

        Args:
            dataset: Dataset to split
            n_folds: Number of folds
            shuffle: Whether to shuffle
            random_state: Random seed

        Returns:
            List of (train, val) tuples for each fold
        """
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2")

        n_samples = len(dataset)
        fold_size = n_samples // n_folds

        logger.info("Creating K-fold splits", n_folds=n_folds, samples_per_fold=fold_size)

        # Shuffle if requested
        indices = np.arange(n_samples)
        if shuffle and random_state is not None:
            np.random.seed(random_state)
            np.random.shuffle(indices)
        elif shuffle:
            np.random.shuffle(indices)

        splits = []
        for fold in range(n_folds):
            # Validation indices for this fold
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples
            val_indices = indices[val_start:val_end]

            # Training indices (everything else)
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

            # Create train and val datasets
            train_data = [dataset[i] for i in train_indices]
            val_data = [dataset[i] for i in val_indices]

            splits.append((train_data, val_data))

        logger.info("K-fold splits created", n_splits=len(splits))
        return splits

    @staticmethod
    def _random_split(
        dataset: List[Dict[str, Any]],
        test_size: float,
        shuffle: bool,
        random_state: Optional[int],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Random split without stratification."""
        n_samples = len(dataset)
        indices = np.arange(n_samples)

        if shuffle:
            if random_state is not None:
                np.random.seed(random_state)
            np.random.shuffle(indices)

        split_idx = int(n_samples * (1.0 - test_size))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train = [dataset[i] for i in train_indices]
        test = [dataset[i] for i in test_indices]

        return train, test

    @staticmethod
    def _stratified_split(
        dataset: List[Dict[str, Any]],
        test_size: float,
        stratify_by: str,
        random_state: Optional[int],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Stratified split maintaining class distribution."""
        # Extract labels for stratification
        labels = []
        for item in dataset:
            # Navigate nested dict structure (e.g., 'label.label')
            value = item
            for key in stratify_by.split("."):
                value = value.get(key, {})
            labels.append(value)

        # Group indices by label
        from collections import defaultdict

        label_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            label_indices[label].append(idx)

        train_indices = []
        test_indices = []

        # Split each class proportionally
        if random_state is not None:
            np.random.seed(random_state)

        for label, indices in label_indices.items():
            indices = np.array(indices)
            np.random.shuffle(indices)

            n_test = max(1, int(len(indices) * test_size))
            test_indices.extend(indices[:n_test])
            train_indices.extend(indices[n_test:])

        train = [dataset[i] for i in train_indices]
        test = [dataset[i] for i in test_indices]

        logger.info(
            "Stratified split complete",
            n_classes=len(label_indices),
            train_size=len(train),
            test_size=len(test),
        )

        return train, test


class DatasetValidator:
    """Validate dataset quality and consistency."""

    @staticmethod
    def validate_dataset(
        dataset: List[Dict[str, Any]], task: str = "qa", strict: bool = False
    ) -> Dict[str, Any]:
        """Validate dataset for training.

        Args:
            dataset: Dataset to validate
            task: Task type (qa, classification, generation, instruction)
            strict: Whether to raise errors on validation failures

        Returns:
            Validation report dictionary
        """
        logger.info("Validating dataset", samples=len(dataset), task=task)

        report = {
            "total_samples": len(dataset),
            "valid_samples": 0,
            "errors": [],
            "warnings": [],
            "statistics": {},
        }

        # Check required fields
        required_fields = ["text"]
        if task in ["qa", "classification", "instruction"]:
            required_fields.append("label")

        valid_count = 0
        text_lengths = []
        empty_text = 0
        missing_labels = 0

        for idx, item in enumerate(dataset):
            is_valid = True

            # Check required fields
            for field in required_fields:
                if field not in item:
                    report["errors"].append(f"Sample {idx}: Missing field '{field}'")
                    is_valid = False

            # Check text content
            if "text" in item:
                text = item["text"]
                if not text or not isinstance(text, str):
                    report["errors"].append(f"Sample {idx}: Invalid or empty text")
                    empty_text += 1
                    is_valid = False
                else:
                    text_lengths.append(len(text))

            # Check label structure based on task
            if "label" in item and item["label"]:
                label = item["label"]
                if task == "qa":
                    if "question" not in label and "answer" not in label:
                        report["warnings"].append(
                            f"Sample {idx}: QA label missing question or answer"
                        )
                elif task == "classification":
                    if "label" not in label:
                        report["warnings"].append(f"Sample {idx}: Classification label missing")
                elif task == "instruction":
                    if "instruction" not in label or "response" not in label:
                        report["warnings"].append(f"Sample {idx}: Instruction label incomplete")
            elif task != "generation":
                missing_labels += 1

            if is_valid:
                valid_count += 1

        report["valid_samples"] = valid_count

        # Calculate statistics
        if text_lengths:
            report["statistics"] = {
                "mean_text_length": np.mean(text_lengths),
                "median_text_length": np.median(text_lengths),
                "min_text_length": np.min(text_lengths),
                "max_text_length": np.max(text_lengths),
                "std_text_length": np.std(text_lengths),
                "empty_texts": empty_text,
                "missing_labels": missing_labels,
            }

        # Summary
        validity_rate = valid_count / len(dataset) if dataset else 0.0
        report["validity_rate"] = validity_rate

        logger.info(
            "Validation complete",
            valid=valid_count,
            total=len(dataset),
            validity_rate=f"{validity_rate:.2%}",
            errors=len(report["errors"]),
            warnings=len(report["warnings"]),
        )

        if strict and report["errors"]:
            raise ValueError(
                f"Dataset validation failed with {len(report['errors'])} errors. "
                f"First error: {report['errors'][0]}"
            )

        return report

    @staticmethod
    def check_class_balance(
        dataset: List[Dict[str, Any]], stratify_by: str = "label.label"
    ) -> Dict[str, Any]:
        """Check class distribution for classification tasks.

        Args:
            dataset: Dataset to check
            stratify_by: Field containing class labels

        Returns:
            Class distribution dictionary
        """
        from collections import Counter

        labels = []
        for item in dataset:
            value = item
            for key in stratify_by.split("."):
                value = value.get(key, {})
            if value:
                labels.append(str(value))

        label_counts = Counter(labels)
        total = len(labels)

        distribution = {
            "total_samples": total,
            "n_classes": len(label_counts),
            "class_counts": dict(label_counts),
            "class_percentages": {
                label: (count / total * 100) for label, count in label_counts.items()
            },
        }

        # Check for imbalance
        if label_counts:
            max_count = max(label_counts.values())
            min_count = min(label_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

            distribution["imbalance_ratio"] = imbalance_ratio
            distribution["is_balanced"] = imbalance_ratio < 3.0

            logger.info(
                "Class distribution",
                n_classes=len(label_counts),
                imbalance_ratio=f"{imbalance_ratio:.2f}",
                balanced=distribution["is_balanced"],
            )

        return distribution


def split_dataset(
    dataset: List[Dict[str, Any]],
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    stratify_by: Optional[str] = None,
    shuffle: bool = True,
    random_state: Optional[int] = 42,
) -> Tuple:
    """Convenience function to split dataset.

    Args:
        dataset: Dataset to split
        test_size: Test set proportion
        val_size: Validation set proportion (if None, only train/test split)
        stratify_by: Field to stratify by
        shuffle: Whether to shuffle
        random_state: Random seed

    Returns:
        Tuple of datasets (train, test) or (train, val, test)
    """
    splitter = DatasetSplitter()

    if val_size is not None:
        return splitter.train_val_test_split(
            dataset, val_size, test_size, stratify_by, shuffle, random_state
        )
    else:
        return splitter.train_test_split(dataset, test_size, stratify_by, shuffle, random_state)


def validate_dataset(
    dataset: List[Dict[str, Any]], task: str = "qa", strict: bool = False
) -> Dict[str, Any]:
    """Convenience function to validate dataset.

    Args:
        dataset: Dataset to validate
        task: Task type
        strict: Whether to raise on errors

    Returns:
        Validation report
    """
    validator = DatasetValidator()
    return validator.validate_dataset(dataset, task, strict)
