"""Model comparison and benchmarking utilities."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from slm_builder.models.evaluation import Evaluator
from slm_builder.utils import get_logger

logger = get_logger(__name__)


class ModelComparator:
    """Compare multiple models on same datasets."""

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize comparator.

        Args:
            output_dir: Directory to save comparison reports
        """
        self.output_dir = Path(output_dir) if output_dir else Path("model_comparisons")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.comparisons = []

    def compare_models(
        self,
        models: List[Tuple[str, Any, Any]],
        dataset: List[Dict[str, Any]],
        metrics: List[str] = None,
        batch_size: int = 8,
    ) -> Dict[str, Any]:
        """Compare multiple models on the same dataset.

        Args:
            models: List of (name, model, tokenizer) tuples
            dataset: Test dataset
            metrics: Metrics to evaluate
            batch_size: Batch size for evaluation

        Returns:
            Comparison results dictionary
        """
        if metrics is None:
            metrics = ["perplexity", "accuracy"]

        logger.info(
            "Starting model comparison",
            n_models=len(models),
            n_samples=len(dataset),
            metrics=metrics,
        )

        results = {
            "timestamp": datetime.now().isoformat(),
            "n_models": len(models),
            "n_samples": len(dataset),
            "metrics": metrics,
            "models": {},
        }

        for model_name, model, tokenizer in models:
            logger.info("Evaluating model", model=model_name)

            start_time = time.time()

            try:
                evaluator = Evaluator(model, tokenizer, device="cuda")
                model_results = evaluator.evaluate(dataset, metrics, batch_size)

                elapsed_time = time.time() - start_time

                results["models"][model_name] = {
                    "metrics": model_results,
                    "evaluation_time": elapsed_time,
                    "samples_per_second": len(dataset) / elapsed_time,
                }

                logger.info(
                    "Model evaluated",
                    model=model_name,
                    time=f"{elapsed_time:.2f}s",
                    perplexity=model_results.get("perplexity"),
                )

            except Exception as e:
                logger.error("Model evaluation failed", model=model_name, error=str(e))
                results["models"][model_name] = {"error": str(e)}

        # Calculate rankings
        results["rankings"] = self._calculate_rankings(results["models"], metrics)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"comparison_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Comparison complete", output=str(output_file))

        self.comparisons.append(results)
        return results

    def _calculate_rankings(
        self, model_results: Dict[str, Any], metrics: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """Calculate model rankings for each metric."""
        rankings = {metric: {} for metric in metrics}

        for metric in metrics:
            # Extract metric values
            metric_values = []
            for model_name, result in model_results.items():
                if "error" not in result and "metrics" in result:
                    value = result["metrics"].get(metric)
                    if value is not None:
                        metric_values.append((model_name, value))

            # Sort based on metric (lower is better for perplexity, higher for accuracy/f1)
            if metric == "perplexity":
                metric_values.sort(key=lambda x: x[1])
            else:
                metric_values.sort(key=lambda x: x[1], reverse=True)

            # Assign ranks
            for rank, (model_name, value) in enumerate(metric_values, 1):
                rankings[metric][model_name] = rank

        return rankings

    def generate_comparison_report(
        self, comparison_results: Dict[str, Any], format: str = "markdown"
    ) -> str:
        """Generate a formatted comparison report.

        Args:
            comparison_results: Results from compare_models
            format: Output format (markdown, html, text)

        Returns:
            Formatted report string
        """
        if format == "markdown":
            return self._generate_markdown_report(comparison_results)
        elif format == "html":
            return self._generate_html_report(comparison_results)
        else:
            return self._generate_text_report(comparison_results)

    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate markdown comparison report."""
        lines = [
            "# Model Comparison Report",
            "",
            f"**Date**: {results['timestamp']}",
            f"**Models**: {results['n_models']}",
            f"**Samples**: {results['n_samples']}",
            "",
            "## Results",
            "",
            "| Model | " + " | ".join(results["metrics"]) + " | Time (s) | Samples/s |",
            "|-------|" + "|".join(["-------"] * (len(results["metrics"]) + 2)) + "|",
        ]

        for model_name, result in results["models"].items():
            if "error" in result:
                row = f"| {model_name} | ERROR | - | - | - |"
            else:
                metrics_values = []
                for metric in results["metrics"]:
                    value = result["metrics"].get(metric, "N/A")
                    if isinstance(value, float):
                        metrics_values.append(f"{value:.4f}")
                    else:
                        metrics_values.append(str(value))

                time_val = result.get("evaluation_time", 0)
                sps_val = result.get("samples_per_second", 0)

                row = (
                    f"| {model_name} | "
                    + " | ".join(metrics_values)
                    + f" | {time_val:.2f} | {sps_val:.2f} |"
                )

            lines.append(row)

        lines.extend(["", "## Rankings", ""])

        for metric, rankings in results.get("rankings", {}).items():
            lines.append(f"### {metric.capitalize()}")
            lines.append("")
            for model_name, rank in sorted(rankings.items(), key=lambda x: x[1]):
                lines.append(f"{rank}. **{model_name}**")
            lines.append("")

        return "\n".join(lines)

    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML comparison report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Model Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        h1, h2 {{ color: #333; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Model Comparison Report</h1>
    <p class="timestamp">Generated: {results['timestamp']}</p>
    <p><strong>Models:</strong> {results['n_models']} | \
<strong>Samples:</strong> {results['n_samples']}</p>

    <h2>Results</h2>
    <table>
        <tr>
            <th>Model</th>
"""
        for metric in results["metrics"]:
            html += f"            <th>{metric.capitalize()}</th>\n"
        html += "            <th>Time (s)</th>\n            <th>Samples/s</th>\n        </tr>\n"

        for model_name, result in results["models"].items():
            html += f"        <tr>\n            <td><strong>{model_name}</strong></td>\n"
            if "error" in result:
                html += "            <td colspan='100'>ERROR</td>\n"
            else:
                for metric in results["metrics"]:
                    value = result["metrics"].get(metric, "N/A")
                    if isinstance(value, float):
                        html += f"            <td>{value:.4f}</td>\n"
                    else:
                        html += f"            <td>{value}</td>\n"
                time_val = result.get("evaluation_time", 0)
                sps_val = result.get("samples_per_second", 0)
                html += f"            <td>{time_val:.2f}</td>\n"
                html += f"            <td>{sps_val:.2f}</td>\n"
            html += "        </tr>\n"

        html += "    </table>\n"

        html += "    <h2>Rankings</h2>\n"
        for metric, rankings in results.get("rankings", {}).items():
            html += f"    <h3>{metric.capitalize()}</h3>\n    <ol>\n"
            for model_name, rank in sorted(rankings.items(), key=lambda x: x[1]):
                html += f"        <li><strong>{model_name}</strong></li>\n"
            html += "    </ol>\n"

        html += "</body>\n</html>"
        return html

    def _generate_text_report(self, results: Dict[str, Any]) -> str:
        """Generate plain text comparison report."""
        lines = [
            "=" * 80,
            "MODEL COMPARISON REPORT",
            "=" * 80,
            f"Date: {results['timestamp']}",
            f"Models: {results['n_models']}",
            f"Samples: {results['n_samples']}",
            "",
            "RESULTS",
            "-" * 80,
        ]

        for model_name, result in results["models"].items():
            lines.append(f"\nModel: {model_name}")
            if "error" in result:
                lines.append(f"  ERROR: {result['error']}")
            else:
                for metric in results["metrics"]:
                    value = result["metrics"].get(metric, "N/A")
                    if isinstance(value, float):
                        lines.append(f"  {metric}: {value:.4f}")
                    else:
                        lines.append(f"  {metric}: {value}")
                time_val = result.get("evaluation_time", 0)
                sps_val = result.get("samples_per_second", 0)
                lines.append(f"  Time: {time_val:.2f}s")
                lines.append(f"  Samples/s: {sps_val:.2f}")

        lines.extend(["", "RANKINGS", "-" * 80])
        for metric, rankings in results.get("rankings", {}).items():
            lines.append(f"\n{metric.capitalize()}:")
            for model_name, rank in sorted(rankings.items(), key=lambda x: x[1]):
                lines.append(f"  {rank}. {model_name}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)


class ExperimentTracker:
    """Track experiments and training runs."""

    def __init__(self, tracking_dir: str = "experiments"):
        """Initialize experiment tracker.

        Args:
            tracking_dir: Directory to store experiment data
        """
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.experiments = []

    def log_experiment(
        self,
        experiment_name: str,
        model_name: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, Any],
        notes: str = "",
    ) -> str:
        """Log an experiment.

        Args:
            experiment_name: Experiment name
            model_name: Model identifier
            hyperparameters: Training hyperparameters
            metrics: Evaluation metrics
            notes: Additional notes

        Returns:
            Experiment ID
        """
        experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        experiment_data = {
            "id": experiment_id,
            "name": experiment_name,
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "notes": notes,
        }

        # Save to file
        exp_file = self.tracking_dir / f"{experiment_id}.json"
        with open(exp_file, "w") as f:
            json.dump(experiment_data, f, indent=2)

        self.experiments.append(experiment_data)

        logger.info("Experiment logged", id=experiment_id, file=str(exp_file))
        return experiment_id

    def list_experiments(self, model_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all tracked experiments.

        Args:
            model_filter: Filter by model name

        Returns:
            List of experiment data
        """
        experiments = []

        for exp_file in self.tracking_dir.glob("*.json"):
            with open(exp_file) as f:
                exp_data = json.load(f)

            if model_filter and exp_data.get("model") != model_filter:
                continue

            experiments.append(exp_data)

        experiments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return experiments

    def get_best_experiment(
        self, metric: str, minimize: bool = True, model_filter: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get best experiment based on a metric.

        Args:
            metric: Metric name
            minimize: Whether to minimize (True) or maximize (False) the metric
            model_filter: Filter by model name

        Returns:
            Best experiment data
        """
        experiments = self.list_experiments(model_filter)

        if not experiments:
            return None

        # Filter experiments that have the metric
        valid_experiments = [exp for exp in experiments if metric in exp.get("metrics", {})]

        if not valid_experiments:
            return None

        # Sort by metric
        valid_experiments.sort(key=lambda x: x["metrics"][metric], reverse=not minimize)

        return valid_experiments[0]


def compare_models(
    models: List[Tuple[str, Any, Any]],
    dataset: List[Dict[str, Any]],
    metrics: List[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function to compare models.

    Args:
        models: List of (name, model, tokenizer) tuples
        dataset: Test dataset
        metrics: Metrics to evaluate
        output_dir: Output directory for reports

    Returns:
        Comparison results
    """
    comparator = ModelComparator(output_dir)
    return comparator.compare_models(models, dataset, metrics)
