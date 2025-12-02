"""Model evaluation and metrics."""

import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from slm_builder.utils import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Model evaluator with multiple metrics."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cpu",
        custom_metrics: Optional[List[Callable]] = None,
    ):
        """Initialize evaluator.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            device: Device
            custom_metrics: Optional list of custom metric functions
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.custom_metrics = custom_metrics or []

        logger.info("Evaluator initialized", device=device)

    def evaluate(
        self,
        dataset: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None,
        batch_size: int = 8,
    ) -> Dict[str, Any]:
        """Evaluate model on dataset.

        Args:
            dataset: Evaluation dataset
            metrics: List of metrics to compute (default: all)
            batch_size: Batch size for evaluation

        Returns:
            Dictionary of metric results
        """
        if metrics is None:
            metrics = ["perplexity", "accuracy", "bleu", "rouge"]

        logger.info("Starting evaluation", samples=len(dataset), metrics=metrics)

        results = {}

        # Compute standard metrics
        if "perplexity" in metrics:
            results["perplexity"] = self.compute_perplexity(dataset, batch_size)

        if "accuracy" in metrics:
            results["accuracy"] = self.compute_accuracy(dataset)

        if "bleu" in metrics:
            results["bleu"] = self.compute_bleu(dataset)

        if "rouge" in metrics:
            results["rouge"] = self.compute_rouge(dataset)

        if "f1" in metrics:
            results["f1"] = self.compute_f1(dataset)

        # Compute custom metrics
        for custom_metric in self.custom_metrics:
            metric_name = custom_metric.__name__
            logger.info("Computing custom metric", metric=metric_name)
            results[metric_name] = custom_metric(self.model, dataset)

        logger.info("Evaluation complete", results=results)
        return results

    def compute_perplexity(self, dataset: List[Dict[str, Any]], batch_size: int = 8) -> float:
        """Compute perplexity on dataset.

        Args:
            dataset: Evaluation dataset
            batch_size: Batch size

        Returns:
            Perplexity score
        """
        try:
            import torch

            self.model.eval()
            total_loss = 0.0
            total_tokens = 0

            with torch.no_grad():
                for i in range(0, len(dataset), batch_size):
                    batch = dataset[i : i + batch_size]
                    texts = [item.get("text", "") for item in batch]

                    # Tokenize
                    inputs = self.tokenizer(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    ).to(self.device)

                    # Forward pass
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss

                    total_loss += loss.item() * inputs["input_ids"].numel()
                    total_tokens += inputs["input_ids"].numel()

            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss)

            logger.info("Perplexity computed", perplexity=perplexity)
            return perplexity

        except Exception as e:
            logger.error("Perplexity computation failed", error=str(e))
            return float("inf")

    def compute_accuracy(self, dataset: List[Dict[str, Any]]) -> float:
        """Compute accuracy for classification/QA tasks.

        Args:
            dataset: Evaluation dataset with labels

        Returns:
            Accuracy score
        """
        correct = 0
        total = 0

        for item in dataset:
            text = item.get("text", "")
            label = item.get("label", {})

            # Get expected answer/label
            expected = None
            if "answer" in label:
                expected = label["answer"]
            elif "label" in label:
                expected = label["label"]
            elif "response" in label:
                expected = label["response"]

            if expected is None:
                continue

            # Generate prediction
            try:
                from slm_builder.models.base import generate_text

                prediction = generate_text(
                    self.model,
                    self.tokenizer,
                    text,
                    max_length=100,
                    device=self.device,
                )

                # Simple exact match (can be improved)
                if expected.lower().strip() in prediction.lower().strip():
                    correct += 1
                total += 1

            except Exception as e:
                logger.warning("Prediction failed", error=str(e))
                continue

        accuracy = correct / total if total > 0 else 0.0
        logger.info("Accuracy computed", accuracy=accuracy, correct=correct, total=total)
        return accuracy

    def compute_bleu(self, dataset: List[Dict[str, Any]]) -> float:
        """Compute BLEU score.

        Args:
            dataset: Evaluation dataset

        Returns:
            BLEU score
        """
        try:
            from nltk.translate.bleu_score import corpus_bleu

            references = []
            hypotheses = []

            for item in dataset:
                text = item.get("text", "")
                label = item.get("label", {})

                # Get reference
                reference = label.get("answer") or label.get("response")
                if not reference:
                    continue

                # Generate hypothesis
                try:
                    from slm_builder.models.base import generate_text

                    hypothesis = generate_text(
                        self.model,
                        self.tokenizer,
                        text,
                        max_length=100,
                        device=self.device,
                    )

                    references.append([reference.split()])
                    hypotheses.append(hypothesis.split())

                except Exception:
                    continue

            if not references:
                return 0.0

            bleu = corpus_bleu(references, hypotheses)
            logger.info("BLEU score computed", bleu=bleu)
            return bleu

        except ImportError:
            logger.warning("NLTK not installed, BLEU score unavailable")
            return 0.0
        except Exception as e:
            logger.error("BLEU computation failed", error=str(e))
            return 0.0

    def compute_rouge(self, dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute ROUGE scores.

        Args:
            dataset: Evaluation dataset

        Returns:
            Dictionary of ROUGE scores
        """
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []

            for item in dataset:
                text = item.get("text", "")
                label = item.get("label", {})

                # Get reference
                reference = label.get("answer") or label.get("response")
                if not reference:
                    continue

                # Generate hypothesis
                try:
                    from slm_builder.models.base import generate_text

                    hypothesis = generate_text(
                        self.model,
                        self.tokenizer,
                        text,
                        max_length=100,
                        device=self.device,
                    )

                    scores = scorer.score(reference, hypothesis)

                    rouge1_scores.append(scores["rouge1"].fmeasure)
                    rouge2_scores.append(scores["rouge2"].fmeasure)
                    rougeL_scores.append(scores["rougeL"].fmeasure)

                except Exception:
                    continue

            results = {
                "rouge1": np.mean(rouge1_scores) if rouge1_scores else 0.0,
                "rouge2": np.mean(rouge2_scores) if rouge2_scores else 0.0,
                "rougeL": np.mean(rougeL_scores) if rougeL_scores else 0.0,
            }

            logger.info("ROUGE scores computed", results=results)
            return results

        except ImportError:
            logger.warning("rouge-score not installed, ROUGE scores unavailable")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        except Exception as e:
            logger.error("ROUGE computation failed", error=str(e))
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def compute_f1(self, dataset: List[Dict[str, Any]]) -> float:
        """Compute F1 score for classification tasks.

        Args:
            dataset: Evaluation dataset

        Returns:
            F1 score
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for item in dataset:
            text = item.get("text", "")
            label = item.get("label", {})

            expected = label.get("label")
            if not expected:
                continue

            # Generate prediction
            try:
                from slm_builder.models.base import generate_text

                prediction = generate_text(
                    self.model,
                    self.tokenizer,
                    text,
                    max_length=50,
                    device=self.device,
                )

                # Simple token-based matching
                expected_tokens = set(expected.lower().split())
                pred_tokens = set(prediction.lower().split())

                tp = len(expected_tokens & pred_tokens)
                fp = len(pred_tokens - expected_tokens)
                fn = len(expected_tokens - pred_tokens)

                true_positives += tp
                false_positives += fp
                false_negatives += fn

            except Exception:
                continue

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        logger.info("F1 score computed", f1=f1, precision=precision, recall=recall)
        return f1


def evaluate_model(
    model: Any,
    tokenizer: Any,
    dataset: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Convenience function to evaluate a model.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        dataset: Evaluation dataset
        metrics: Metrics to compute
        device: Device

    Returns:
        Dictionary of results
    """
    evaluator = Evaluator(model, tokenizer, device)
    return evaluator.evaluate(dataset, metrics=metrics)
