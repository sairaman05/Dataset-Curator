# core/model_analyzer.py
"""
Analyzes the dataset and recommends the best models based on data characteristics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

from core.model_registry import get_registry, RegisteredModel
from utils.helpers import detect_task_type


@dataclass
class DataAnalysis:
    """Results of analyzing a dataset for ML readiness."""
    task_type: str                     # binary_classification / multiclass_classification / regression
    target_column: str
    n_samples: int
    n_features: int
    n_numeric: int
    n_categorical: int
    n_classes: int                     # 0 for regression
    class_distribution: Dict[str, int] # Empty for regression
    has_imbalance: bool
    has_high_dimensionality: bool      # features > 50
    has_outliers: bool
    outlier_pct: float
    has_missing: bool
    missing_pct: float
    feature_names: List[str]
    target_dtype: str
    dataset_size_category: str         # "small" (<1K), "medium" (1K-50K), "large" (>50K)


@dataclass
class ModelRecommendation:
    """A scored model recommendation."""
    model_info: RegisteredModel
    score: float                       # 0-100
    rank: int
    reasons: List[str]
    warnings: List[str]


class ModelAnalyzer:
    """Analyzes data characteristics and recommends suitable models."""

    def __init__(self):
        self.registry = get_registry()

    def analyze_data(self, df: pd.DataFrame, target_column: str) -> DataAnalysis:
        """Analyze dataset characteristics relevant to model selection."""
        target = df[target_column]
        features = df.drop(columns=[target_column])

        task_type, n_classes = detect_task_type(target)

        n_numeric = len(features.select_dtypes(include=[np.number]).columns)
        n_categorical = len(features.select_dtypes(include=["object", "category"]).columns)

        # Class distribution
        class_dist = {}
        has_imbalance = False
        if "classification" in task_type:
            vc = target.value_counts()
            class_dist = vc.to_dict()
            if len(vc) >= 2:
                ratio = vc.iloc[-1] / vc.iloc[0]
                has_imbalance = ratio < 0.3

        # Outliers (IQR on numeric)
        numeric_cols = features.select_dtypes(include=[np.number])
        outlier_counts = 0
        total_vals = 0
        for col in numeric_cols.columns:
            q1, q3 = numeric_cols[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            outlier_counts += ((numeric_cols[col] < q1 - 1.5 * iqr) | (numeric_cols[col] > q3 + 1.5 * iqr)).sum()
            total_vals += len(numeric_cols[col])
        outlier_pct = (outlier_counts / total_vals * 100) if total_vals > 0 else 0

        # Missing
        missing_total = features.isnull().sum().sum()
        missing_pct = (missing_total / (features.shape[0] * features.shape[1]) * 100) if features.size > 0 else 0

        # Size category
        n = len(df)
        if n < 1000:
            size_cat = "small"
        elif n < 50000:
            size_cat = "medium"
        else:
            size_cat = "large"

        return DataAnalysis(
            task_type=task_type,
            target_column=target_column,
            n_samples=len(df),
            n_features=len(features.columns),
            n_numeric=n_numeric,
            n_categorical=n_categorical,
            n_classes=n_classes if "classification" in task_type else 0,
            class_distribution={str(k): int(v) for k, v in class_dist.items()},
            has_imbalance=has_imbalance,
            has_high_dimensionality=len(features.columns) > 50,
            has_outliers=outlier_pct > 5,
            outlier_pct=round(outlier_pct, 2),
            has_missing=missing_pct > 0,
            missing_pct=round(missing_pct, 2),
            feature_names=list(features.columns),
            target_dtype=str(target.dtype),
            dataset_size_category=size_cat,
        )

    def recommend_models(self, analysis: DataAnalysis, top_n: int = 5) -> List[ModelRecommendation]:
        """Score and rank all available models for the detected task."""
        available = self.registry.get_models_for_task(analysis.task_type)
        scored = []

        for model in available:
            score, reasons, warnings = self._score_model(model, analysis)
            scored.append((model, score, reasons, warnings))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        recommendations = []
        for rank, (model, score, reasons, warnings) in enumerate(scored, 1):
            recommendations.append(ModelRecommendation(
                model_info=model,
                score=round(score, 1),
                rank=rank,
                reasons=reasons,
                warnings=warnings,
            ))

        return recommendations[:top_n] if top_n else recommendations

    def get_all_models_for_task(self, analysis: DataAnalysis) -> List[ModelRecommendation]:
        """Return ALL models (not just top N) with scores."""
        return self.recommend_models(analysis, top_n=0)

    def _score_model(self, model: RegisteredModel, analysis: DataAnalysis) -> Tuple[float, List[str], List[str]]:
        """Score a model 0-100 based on data characteristics."""
        score = 50.0
        reasons = []
        warnings = []

        name = model.class_name
        n = analysis.n_samples
        p = analysis.n_features

        # ── Dataset size scoring ──
        if analysis.dataset_size_category == "small":
            if name in ("LogisticRegression", "Ridge", "Lasso", "LinearRegression",
                        "KNeighborsClassifier", "KNeighborsRegressor",
                        "DecisionTreeClassifier", "DecisionTreeRegressor",
                        "GaussianNB", "BernoulliNB", "SVC", "SVR"):
                score += 10
                reasons.append("Good for small datasets — lower risk of overfitting")
            if name in ("XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor"):
                score -= 5
                warnings.append("Boosting methods may overfit on small datasets")

        elif analysis.dataset_size_category == "medium":
            if name in ("RandomForestClassifier", "RandomForestRegressor",
                        "GradientBoostingClassifier", "GradientBoostingRegressor",
                        "XGBClassifier", "XGBRegressor"):
                score += 10
                reasons.append("Well-suited for medium-sized datasets")

        elif analysis.dataset_size_category == "large":
            if name in ("SGDClassifier", "SGDRegressor",
                        "HistGradientBoostingClassifier", "HistGradientBoostingRegressor",
                        "LGBMClassifier", "LGBMRegressor",
                        "XGBClassifier", "XGBRegressor"):
                score += 15
                reasons.append("Scales efficiently to large datasets")
            if name in ("SVC", "SVR", "KNeighborsClassifier", "KNeighborsRegressor"):
                score -= 15
                warnings.append("Very slow on large datasets (O(n²) or O(n³) complexity)")

        # ── High dimensionality ──
        if analysis.has_high_dimensionality:
            if name in ("Lasso", "ElasticNet", "SGDClassifier", "SGDRegressor",
                        "RandomForestClassifier", "RandomForestRegressor"):
                score += 8
                reasons.append("Handles high-dimensional data well")
            if name in ("KNeighborsClassifier", "KNeighborsRegressor"):
                score -= 10
                warnings.append("Curse of dimensionality degrades KNN performance")

        # ── Class imbalance ──
        if analysis.has_imbalance:
            if name in ("RandomForestClassifier", "GradientBoostingClassifier",
                        "XGBClassifier", "LGBMClassifier",
                        "ExtraTreesClassifier", "HistGradientBoostingClassifier"):
                score += 8
                reasons.append("Ensemble methods handle class imbalance better")
            if name in ("LogisticRegression", "GaussianNB"):
                score -= 5
                warnings.append("May struggle with imbalanced classes without resampling")

        # ── Outliers ──
        if analysis.has_outliers:
            if name in ("RandomForestClassifier", "RandomForestRegressor",
                        "GradientBoostingClassifier", "GradientBoostingRegressor",
                        "XGBClassifier", "XGBRegressor",
                        "DecisionTreeClassifier", "DecisionTreeRegressor",
                        "ExtraTreesClassifier", "ExtraTreesRegressor"):
                score += 5
                reasons.append("Tree-based models are robust to outliers")
            if name in ("LinearRegression", "Ridge", "Lasso",
                        "LogisticRegression", "SVR", "SVC"):
                score -= 5
                warnings.append("Linear/SVM models are sensitive to outliers")

        # ── Ensemble bonus ──
        if name in ("RandomForestClassifier", "RandomForestRegressor",
                     "GradientBoostingClassifier", "GradientBoostingRegressor",
                     "XGBClassifier", "XGBRegressor",
                     "LGBMClassifier", "LGBMRegressor",
                     "ExtraTreesClassifier", "ExtraTreesRegressor",
                     "HistGradientBoostingClassifier", "HistGradientBoostingRegressor"):
            score += 5
            reasons.append("Ensemble method — generally strong out-of-box performance")

        # ── Interpretability bonus for simple models ──
        if name in ("LogisticRegression", "LinearRegression", "Ridge", "Lasso",
                     "DecisionTreeClassifier", "DecisionTreeRegressor"):
            score += 3
            reasons.append("Highly interpretable model")

        # ── Mixed feature types ──
        if analysis.n_categorical > 0 and analysis.n_numeric > 0:
            if name in ("RandomForestClassifier", "RandomForestRegressor",
                        "GradientBoostingClassifier", "GradientBoostingRegressor",
                        "XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor",
                        "HistGradientBoostingClassifier", "HistGradientBoostingRegressor"):
                score += 5
                reasons.append("Handles mixed feature types naturally")

        # ── Clamp ──
        score = max(5, min(100, score))

        if not reasons:
            reasons.append("Standard model suitable for this task")

        return score, reasons, warnings