# core/model_comparator.py
"""
Side-by-side model comparison with detailed analysis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

from core.model_trainer import TrainingResult


@dataclass
class ComparisonResult:
    """Result of comparing two or more models."""
    results: List[TrainingResult]
    winner: str                                # display_name of best model
    winner_reason: str
    metric_comparison: Dict[str, Dict[str, float]]  # metric -> {model_name: value}
    training_time_comparison: Dict[str, float]
    summary: str


class ModelComparator:
    """Compare trained models side-by-side."""

    def compare(self, results: List[TrainingResult]) -> ComparisonResult:
        """Compare two or more trained models."""
        if len(results) < 2:
            raise ValueError("Need at least 2 models to compare.")

        task_type = results[0].task_type
        is_clf = "classification" in task_type

        # Build metric comparison
        all_metrics = set()
        for r in results:
            all_metrics.update(r.metrics.keys())

        metric_comparison = {}
        for metric in sorted(all_metrics):
            metric_comparison[metric] = {}
            for r in results:
                metric_comparison[metric][r.display_name] = r.metrics.get(metric, None)

        # Training times
        time_comparison = {r.display_name: round(r.total_time, 2) for r in results}

        # Determine winner
        primary_metric = "accuracy" if is_clf else "r2"
        best_result = max(results, key=lambda r: r.metrics.get(primary_metric, 0))
        winner = best_result.display_name

        # Build reason
        best_val = best_result.metrics.get(primary_metric, 0)
        metric_label = "Accuracy" if is_clf else "R²"
        winner_reason = f"Highest {metric_label}: {best_val:.4f}"

        # Check if runner-up is close
        sorted_by_primary = sorted(results, key=lambda r: r.metrics.get(primary_metric, 0), reverse=True)
        if len(sorted_by_primary) >= 2:
            runner_up = sorted_by_primary[1]
            diff = best_val - runner_up.metrics.get(primary_metric, 0)
            if diff < 0.01:
                winner_reason += f" (marginally better than {runner_up.display_name} by {diff:.4f})"

        # Summary
        summary = self._build_summary(results, metric_comparison, winner, task_type)

        return ComparisonResult(
            results=results,
            winner=winner,
            winner_reason=winner_reason,
            metric_comparison=metric_comparison,
            training_time_comparison=time_comparison,
            summary=summary,
        )

    def _build_summary(self, results, metric_comparison, winner, task_type) -> str:
        """Build a text summary of the comparison."""
        lines = []
        lines.append(f"## Model Comparison Summary\n")
        lines.append(f"**Models compared:** {', '.join(r.display_name for r in results)}")
        lines.append(f"**Task:** {task_type.replace('_', ' ').title()}")
        lines.append(f"**Winner:** 🏆 {winner}\n")

        lines.append("### Metrics")
        for metric, values in metric_comparison.items():
            lines.append(f"\n**{metric.upper()}:**")
            for name, val in values.items():
                if val is not None:
                    flag = " ← Best" if name == winner else ""
                    lines.append(f"  - {name}: {val:.4f}{flag}")

        lines.append(f"\n### Training Time")
        for r in results:
            lines.append(f"  - {r.display_name}: {r.total_time:.2f}s")

        fastest = min(results, key=lambda r: r.total_time)
        lines.append(f"\n**Fastest:** {fastest.display_name} ({fastest.total_time:.2f}s)")

        return "\n".join(lines)