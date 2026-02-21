# agents/ml_agent.py
"""
ML Agent — Orchestrates the entire ML pipeline:
  1. Analyze data → recommend models
  2. Train selected model(s) with live progress
  3. Evaluate and generate reports
  4. Compare models side-by-side
"""

from typing import List, Optional, Callable, Dict, Any
import pandas as pd
import numpy as np

from core.model_registry import get_registry, ModelRegistry
from core.model_analyzer import ModelAnalyzer, DataAnalysis, ModelRecommendation
from core.model_trainer import ModelTrainer, TrainingResult, ProgressCallback
from core.model_evaluator import *
from core.model_comparator import ModelComparator, ComparisonResult
from core.report_generator import ReportGenerator
from utils.helpers import nuke_datetime_columns


class MLAgent:
    """
    Orchestrates the ML pipeline.

    Usage:
        agent = MLAgent()
        analysis = agent.analyze(df, "target_col")
        recommendations = agent.recommend(analysis)
        all_models = agent.get_all_models(analysis)
        result = agent.train("RandomForestClassifier", df, "target_col", analysis, epochs=50, callback=my_fn)
        comparison = agent.compare([result1, result2])
        report_md = agent.generate_report(result)
        comparison_md = agent.generate_comparison_report(comparison)
    """

    def __init__(self):
        self.registry: ModelRegistry = get_registry()
        self.analyzer = ModelAnalyzer()
        self.trainer = ModelTrainer()
        self.comparator = ModelComparator()
        self.report_gen = ReportGenerator()

        # Cache prepared data to avoid re-processing for multiple model trains
        self._cached_data = None
        self._cached_target = None
        self._cached_task_type = None

    def analyze(self, df: pd.DataFrame, target_column: str) -> DataAnalysis:
        """Analyze dataset for ML readiness."""
        # Pre-process: nuke datetimes
        clean_df = nuke_datetime_columns(df.copy())
        # If target was a datetime column and got transformed, update name
        if target_column not in clean_df.columns:
            # Check if any derived columns exist
            derived = [c for c in clean_df.columns if c.startswith(target_column)]
            if derived:
                target_column = derived[0]  # Use first derived column
            else:
                raise ValueError(f"Target column '{target_column}' not found after preprocessing.")

        return self.analyzer.analyze_data(clean_df, target_column)

    def recommend(self, analysis: DataAnalysis, top_n: int = 5) -> List[ModelRecommendation]:
        """Get top N model recommendations."""
        return self.analyzer.recommend_models(analysis, top_n=top_n)

    def get_all_models(self, analysis: DataAnalysis) -> List[ModelRecommendation]:
        """Get ALL available models for the task with scores."""
        return self.analyzer.get_all_models_for_task(analysis)

    def train(
        self,
        model_class_name: str,
        df: pd.DataFrame,
        target_column: str,
        analysis: DataAnalysis,
        epochs: int = 50,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TrainingResult:
        """Train a single model."""
        # Prepare data (cache it)
        if (self._cached_data is None or
            self._cached_target != target_column or
            self._cached_task_type != analysis.task_type):

            clean_df = nuke_datetime_columns(df.copy())
            if target_column not in clean_df.columns:
                derived = [c for c in clean_df.columns if c.startswith(target_column)]
                if derived:
                    target_column = derived[0]

            prepared = self.trainer.prepare_data(clean_df, target_column, analysis.task_type)
            self._cached_data = prepared
            self._cached_target = target_column
            self._cached_task_type = analysis.task_type

        X_train, X_val, X_test, y_train, y_val, y_test, \
            feature_names, label_encoder, class_names, scaler = self._cached_data

        return self.trainer.train(
            model_class_name=model_class_name,
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            task_type=analysis.task_type,
            feature_names=feature_names,
            label_encoder=label_encoder,
            class_names=class_names,
            scaler=scaler,
            epochs=epochs,
            progress_callback=progress_callback,
        )

    def compare(self, results: List[TrainingResult]) -> ComparisonResult:
        """Compare two or more trained models."""
        return self.comparator.compare(results)

    def generate_report(self, result: TrainingResult) -> str:
        """Generate markdown report for a single model."""
        return self.report_gen.generate_single_model_report(result)

    def generate_comparison_report(self, comparison: ComparisonResult) -> str:
        """Generate markdown report for model comparison."""
        return self.report_gen.generate_comparison_report(comparison)

    def clear_cache(self):
        """Clear cached prepared data (call on new file upload)."""
        self._cached_data = None
        self._cached_target = None
        self._cached_task_type = None