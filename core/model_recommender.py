"""
Model Recommender — AI-powered model selection using Gemini 2.5 Flash.

Analyzes actual data characteristics and sends a structured summary to Gemini
to get truly data-specific top 3-4 recommendations with reasoning.
Falls back to heuristic scoring if API is unavailable.
"""

import pandas as pd
import numpy as np
import json
import re
import os
from dataclasses import dataclass, field
from enum import Enum
from config.settings import CLASSIFICATION_UNIQUE_THRESHOLD


class TaskType(Enum):
    BINARY_CLASSIFICATION = "Binary Classification"
    MULTICLASS_CLASSIFICATION = "Multiclass Classification"
    REGRESSION = "Regression"


@dataclass
class ModelRecommendation:
    name: str
    model_key: str
    rank: int
    task_type: TaskType
    reasoning: list = field(default_factory=list)
    strengths: list = field(default_factory=list)
    weaknesses: list = field(default_factory=list)
    is_pretrained: bool = False
    pretrained_note: str = ""
    estimated_train_time: str = "Fast"
    complexity: str = "Low"
    suggested_epochs: int = 50
    suggested_hyperparams: dict = field(default_factory=dict)
    architecture_description: str = ""


@dataclass
class DataAnalysis:
    task_type: TaskType
    n_samples: int
    n_features: int
    n_numeric: int
    n_categorical: int
    n_classes: int = 0
    class_names: list = field(default_factory=list)
    class_distribution: dict = field(default_factory=dict)
    is_imbalanced: bool = False
    has_high_dimensionality: bool = False
    has_outliers: bool = False
    has_strong_correlations: bool = False
    has_missing_values: bool = False
    outlier_columns: list = field(default_factory=list)
    correlated_pairs: list = field(default_factory=list)
    skewed_columns: list = field(default_factory=list)
    target_column: str = ""
    feature_columns: list = field(default_factory=list)
    data_summary: str = ""


# ══════════════════════════════════════════════════════════════
# COMPREHENSIVE MODEL CATALOG — sklearn + basic DL
# ══════════════════════════════════════════════════════════════
MODEL_CATALOG = {
    # ── Classification: sklearn ──
    "logistic_regression": {
        "name": "Logistic Regression",
        "type": "classification",
        "complexity": "Low",
        "time": "Very Fast",
        "category": "sklearn",
        "architecture": "Linear model with sigmoid activation. Uses regularization (L1/L2) to prevent overfitting.",
        "strengths": ["Highly interpretable", "Fast training", "Calibrated probabilities", "Works well with linearly separable data"],
        "weaknesses": ["Cannot capture non-linear boundaries", "Sensitive to outliers", "May underfit complex patterns"],
    },
    "random_forest_clf": {
        "name": "Random Forest Classifier",
        "type": "classification",
        "complexity": "Medium",
        "time": "Moderate",
        "category": "sklearn",
        "architecture": "Ensemble of decision trees trained on random subsets. Each tree votes, majority wins. Uses bootstrap aggregating (bagging).",
        "strengths": ["Handles non-linear relationships", "Robust to outliers", "Built-in feature importance", "No scaling needed"],
        "weaknesses": ["Can overfit on noisy data", "Higher memory usage", "Less interpretable"],
    },
    "xgboost_clf": {
        "name": "XGBoost Classifier",
        "type": "classification",
        "complexity": "Medium",
        "time": "Moderate",
        "category": "sklearn",
        "architecture": "Gradient boosting with tree-based learners. Sequential training where each tree corrects previous errors. Uses regularization and tree pruning.",
        "strengths": ["State-of-the-art for tabular data", "Built-in L1/L2 regularization", "Handles missing values", "Excellent with mixed features"],
        "weaknesses": ["More hyperparameters", "Can overfit small datasets", "Slower than LightGBM on large data"],
    },
    "lightgbm_clf": {
        "name": "LightGBM Classifier",
        "type": "classification",
        "complexity": "Medium",
        "time": "Fast",
        "category": "sklearn",
        "architecture": "Gradient boosting using histogram-based learning. Leaf-wise tree growth (vs level-wise). Optimized for speed and memory efficiency.",
        "strengths": ["Fastest gradient boosting", "Memory efficient", "Native categorical support", "Excellent accuracy"],
        "weaknesses": ["Can overfit small datasets (<1000 rows)", "Sensitive to hyperparameters", "Leaf-wise growth instability"],
    },
    "svm_clf": {
        "name": "Support Vector Machine",
        "type": "classification",
        "complexity": "High",
        "time": "Slow for large data",
        "category": "sklearn",
        "architecture": "Finds optimal hyperplane that maximizes margin between classes. Kernel trick maps data to higher dimensions for non-linear separation.",
        "strengths": ["Effective in high-dimensional spaces", "Kernel trick for non-linear boundaries", "Clear margin of separation"],
        "weaknesses": ["Very slow on >10K rows", "Sensitive to scaling", "Memory intensive"],
    },
    "knn_clf": {
        "name": "K-Nearest Neighbors",
        "type": "classification",
        "complexity": "Low",
        "time": "Fast train, slow predict",
        "category": "sklearn",
        "architecture": "Instance-based learning. Classifies based on majority vote of K nearest neighbors using distance metrics (Euclidean, Manhattan).",
        "strengths": ["Simple and intuitive", "No training phase", "Handles multi-class naturally"],
        "weaknesses": ["Slow prediction on large data", "Sensitive to scaling", "Curse of dimensionality"],
    },
    "extra_trees_clf": {
        "name": "Extra Trees Classifier",
        "type": "classification",
        "complexity": "Medium",
        "time": "Fast",
        "category": "sklearn",
        "architecture": "Similar to Random Forest but uses random thresholds for splits instead of optimal. Faster and more randomized.",
        "strengths": ["Faster than Random Forest", "More randomization reduces overfitting", "Good for high-dimensional data"],
        "weaknesses": ["Less interpretable", "May underfit with too few trees", "Higher variance"],
    },
    "gradient_boosting_clf": {
        "name": "Gradient Boosting Classifier",
        "type": "classification",
        "complexity": "Medium",
        "time": "Moderate",
        "category": "sklearn",
        "architecture": "Sequential ensemble where each tree fits residual errors. Uses gradient descent to minimize loss function.",
        "strengths": ["Strong performance on structured data", "Sequential error correction", "Built-in feature importance"],
        "weaknesses": ["Slower than XGBoost/LightGBM", "Prone to overfitting without tuning", "No parallelism"],
    },
    "ada_boost_clf": {
        "name": "AdaBoost Classifier",
        "type": "classification",
        "complexity": "Low",
        "time": "Fast",
        "category": "sklearn",
        "architecture": "Adaptive boosting that reweights misclassified samples. Weak learners (decision stumps) combined into strong classifier.",
        "strengths": ["Focuses on hard-to-classify samples", "Less prone to overfitting", "Simple to implement"],
        "weaknesses": ["Sensitive to noisy data", "Weak with many features", "Performance ceiling lower than XGBoost"],
    },
    "naive_bayes_clf": {
        "name": "Naive Bayes (Gaussian)",
        "type": "classification",
        "complexity": "Low",
        "time": "Very Fast",
        "category": "sklearn",
        "architecture": "Probabilistic classifier using Bayes theorem. Assumes feature independence (naive assumption). Gaussian distribution for continuous features.",
        "strengths": ["Extremely fast", "Works well with small data", "Good for text classification", "Handles streaming data"],
        "weaknesses": ["Independence assumption often violated", "Sensitive to irrelevant features", "Poor probability estimates"],
    },
    "decision_tree_clf": {
        "name": "Decision Tree Classifier",
        "type": "classification",
        "complexity": "Low",
        "time": "Fast",
        "category": "sklearn",
        "architecture": "Tree structure with decision nodes based on feature thresholds. Recursive binary splitting until stopping criteria met.",
        "strengths": ["Highly interpretable", "No scaling needed", "Handles non-linear relationships", "Fast prediction"],
        "weaknesses": ["Prone to overfitting", "Unstable (small data changes affect structure)", "Biased toward dominant classes"],
    },
    
    # ── Regression: sklearn ──
    "linear_regression": {
        "name": "Linear Regression",
        "type": "regression",
        "complexity": "Low",
        "time": "Very Fast",
        "category": "sklearn",
        "architecture": "Fits linear equation y = w₁x₁ + w₂x₂ + ... + b. Minimizes mean squared error using ordinary least squares (OLS).",
        "strengths": ["Most interpretable", "Clear coefficients", "Fast baseline", "Statistical inference"],
        "weaknesses": ["Assumes linear relationships", "Sensitive to outliers", "Cannot capture complexity"],
    },
    "ridge_regression": {
        "name": "Ridge Regression (L2)",
        "type": "regression",
        "complexity": "Low",
        "time": "Very Fast",
        "category": "sklearn",
        "architecture": "Linear regression with L2 regularization (penalty on coefficient magnitude). Prevents overfitting by shrinking coefficients.",
        "strengths": ["Prevents overfitting via L2", "Handles multicollinearity", "Stable with correlated features"],
        "weaknesses": ["Assumes linear relationships", "No feature selection", "Sensitive to scaling"],
    },
    "lasso_regression": {
        "name": "Lasso Regression (L1)",
        "type": "regression",
        "complexity": "Low",
        "time": "Very Fast",
        "category": "sklearn",
        "architecture": "Linear regression with L1 regularization. Penalty forces some coefficients to exactly zero, performing automatic feature selection.",
        "strengths": ["Built-in feature selection", "L1 regularization zeros out irrelevant features", "Interpretable sparse models"],
        "weaknesses": ["Assumes linearity", "Unstable with correlated features", "May under-select features"],
    },
    "random_forest_reg": {
        "name": "Random Forest Regressor",
        "type": "regression",
        "complexity": "Medium",
        "time": "Moderate",
        "category": "sklearn",
        "architecture": "Ensemble of regression trees. Each tree predicts, final prediction is average. Bootstrap samples and random feature subsets.",
        "strengths": ["Captures non-linearity", "Robust to outliers", "No scaling needed", "Feature importance"],
        "weaknesses": ["Cannot extrapolate", "Higher memory", "Less interpretable"],
    },
    "xgboost_reg": {
        "name": "XGBoost Regressor",
        "type": "regression",
        "complexity": "Medium",
        "time": "Moderate",
        "category": "sklearn",
        "architecture": "Gradient boosting for regression. Sequential tree building to minimize MSE. Regularization and early stopping prevent overfitting.",
        "strengths": ["Top tabular performer", "Handles missing data", "Built-in regularization", "Feature importance"],
        "weaknesses": ["Complex to tune", "Cannot extrapolate", "Risk of overfitting small data"],
    },
    "lightgbm_reg": {
        "name": "LightGBM Regressor",
        "type": "regression",
        "complexity": "Medium",
        "time": "Fast",
        "category": "sklearn",
        "architecture": "Fast gradient boosting using histogram-based splits. Leaf-wise growth for better accuracy. Memory-efficient.",
        "strengths": ["Very fast training", "Memory efficient", "Native categorical support"],
        "weaknesses": ["Can overfit small data", "Sensitive to hyperparameters", "Less stable than XGBoost"],
    },
    "svr": {
        "name": "Support Vector Regressor",
        "type": "regression",
        "complexity": "High",
        "time": "Slow for large data",
        "category": "sklearn",
        "architecture": "SVM adapted for regression. Uses epsilon-insensitive loss (ignore errors within epsilon tube). Kernel trick for non-linearity.",
        "strengths": ["Good for small-medium datasets", "Kernel trick", "Robust with epsilon-insensitive loss"],
        "weaknesses": ["Very slow on large data", "Needs scaling", "Hard to interpret"],
    },
    "extra_trees_reg": {
        "name": "Extra Trees Regressor",
        "type": "regression",
        "complexity": "Medium",
        "time": "Fast",
        "category": "sklearn",
        "architecture": "Extremely randomized trees. Random splits instead of best splits. Faster than Random Forest.",
        "strengths": ["Faster than Random Forest", "Reduced overfitting", "Good generalization"],
        "weaknesses": ["Cannot extrapolate", "Less interpretable", "Higher variance"],
    },
    "gradient_boosting_reg": {
        "name": "Gradient Boosting Regressor",
        "type": "regression",
        "complexity": "Medium",
        "time": "Moderate",
        "category": "sklearn",
        "architecture": "Sequential boosting for regression. Each tree fits residuals. Gradient descent minimizes loss.",
        "strengths": ["Strong structured-data performance", "Sequential error correction", "Feature importance"],
        "weaknesses": ["Slower than LightGBM", "Overfitting risk", "No parallelism"],
    },
    "decision_tree_reg": {
        "name": "Decision Tree Regressor",
        "type": "regression",
        "complexity": "Low",
        "time": "Fast",
        "category": "sklearn",
        "architecture": "Tree structure for regression. Splits minimize MSE at each node. Prediction is mean of leaf samples.",
        "strengths": ["Interpretable", "No scaling needed", "Fast", "Handles non-linearity"],
        "weaknesses": ["Overfits easily", "Unstable", "Cannot extrapolate"],
    },
    "elastic_net": {
        "name": "Elastic Net",
        "type": "regression",
        "complexity": "Low",
        "time": "Very Fast",
        "category": "sklearn",
        "architecture": "Combines L1 (Lasso) and L2 (Ridge) regularization. Balances feature selection and coefficient shrinkage.",
        "strengths": ["Best of Lasso + Ridge", "Handles correlated features better than Lasso", "Feature selection"],
        "weaknesses": ["Still assumes linearity", "Two hyperparameters to tune", "Sensitive to scaling"],
    },
    
    # ── Neural Networks (basic) ──
    "mlp_clf": {
        "name": "Multi-Layer Perceptron Classifier",
        "type": "classification",
        "complexity": "High",
        "time": "Moderate",
        "category": "neural",
        "architecture": "Feedforward neural network with hidden layers. ReLU activation, softmax output. Backpropagation with Adam optimizer.",
        "strengths": ["Can learn complex patterns", "Flexible architecture", "Good with large data"],
        "weaknesses": ["Needs scaling", "Prone to overfitting", "Requires tuning", "Black box"],
    },
    "mlp_reg": {
        "name": "Multi-Layer Perceptron Regressor",
        "type": "regression",
        "complexity": "High",
        "time": "Moderate",
        "category": "neural",
        "architecture": "Feedforward neural network for regression. Hidden layers with ReLU. Linear output layer. MSE loss with Adam optimizer.",
        "strengths": ["Learns non-linear patterns", "Flexible", "Good with large data"],
        "weaknesses": ["Needs scaling", "Overfitting risk", "Hyperparameter sensitive", "Black box"],
    },
}


class ModelRecommender:
    """
    AI-powered model recommender using Gemini 2.5 Flash.
    Falls back to heuristic scoring if the API call fails.
    """

    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target_column = target_column
        self.analysis: DataAnalysis = None

    # ──────────────────────────────────────────────
    # DEEP DATA ANALYSIS
    # ──────────────────────────────────────────────
    def analyze(self) -> DataAnalysis:
        target = self.df[self.target_column]
        features = self.df.drop(columns=[self.target_column])
        n_samples = len(self.df)
        n_unique = target.nunique()

        # Task type
        if target.dtype in ["object", "category"] or (
            pd.api.types.is_numeric_dtype(target)
            and n_unique <= CLASSIFICATION_UNIQUE_THRESHOLD
        ):
            task_type = (
                TaskType.BINARY_CLASSIFICATION
                if n_unique == 2
                else TaskType.MULTICLASS_CLASSIFICATION
            )
        else:
            task_type = TaskType.REGRESSION

        # Class distribution
        class_names, class_distribution, is_imbalanced, n_classes = [], {}, False, 0
        if task_type != TaskType.REGRESSION:
            counts = target.value_counts(normalize=True)
            class_names = [str(c) for c in counts.index.tolist()]
            class_distribution = {str(k): round(v, 4) for k, v in counts.items()}
            n_classes = len(class_names)
            is_imbalanced = counts.iloc[0] > 0.65

        # Features
        numeric_features = features.select_dtypes(include=["number"])
        cat_features = features.select_dtypes(include=["object", "category"])

        # Outliers
        outlier_columns = []
        for col in numeric_features.columns:
            q1, q3 = numeric_features[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                pct = (
                    (numeric_features[col] < q1 - 1.5 * iqr)
                    | (numeric_features[col] > q3 + 1.5 * iqr)
                ).sum() / n_samples
                if pct > 0.03:
                    outlier_columns.append(f"{col} ({pct:.1%} outliers)")

        # Correlations
        correlated_pairs = []
        if numeric_features.shape[1] > 1:
            corr = numeric_features.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            for c in upper.columns:
                for idx in upper.index:
                    v = upper.loc[idx, c]
                    if pd.notna(v) and v > 0.7:
                        correlated_pairs.append(f"{idx} ↔ {c} (r={v:.2f})")

        # Skewness
        skewed_columns = []
        for col in numeric_features.columns:
            sk = numeric_features[col].skew()
            if abs(sk) > 1.0:
                skewed_columns.append(
                    f"{col} (skew={sk:.2f}, {'right' if sk > 0 else 'left'})"
                )

        has_missing = features.isnull().any().any()

        # Build text summary for LLM
        if task_type == TaskType.REGRESSION:
            target_stats = (
                f"Target '{self.target_column}': mean={target.mean():.2f}, "
                f"std={target.std():.2f}, min={target.min():.2f}, "
                f"max={target.max():.2f}, skew={target.skew():.2f}"
            )
        else:
            target_stats = (
                f"Target '{self.target_column}': {n_classes} classes, "
                f"distribution={class_distribution}"
            )

        feature_dtypes = {}
        for col in features.columns:
            dt = str(features[col].dtype)
            feature_dtypes[dt] = feature_dtypes.get(dt, 0) + 1

        data_summary = (
            f"Dataset: {n_samples} rows, {features.shape[1]} features\n"
            f"Task: {task_type.value}\n"
            f"{target_stats}\n"
            f"Feature types: {feature_dtypes}\n"
            f"Numeric: {numeric_features.shape[1]}, Categorical: {cat_features.shape[1]}\n"
            f"Imbalanced: {is_imbalanced}\n"
            f"Missing values: {has_missing}\n"
            f"Outlier columns: {outlier_columns[:5] if outlier_columns else 'None'}\n"
            f"Strong correlations: {correlated_pairs[:5] if correlated_pairs else 'None'}\n"
            f"Skewed columns: {skewed_columns[:5] if skewed_columns else 'None'}\n"
            f"High dimensionality: {features.shape[1] > 50}"
        )

        self.analysis = DataAnalysis(
            task_type=task_type,
            n_samples=n_samples,
            n_features=features.shape[1],
            n_numeric=numeric_features.shape[1],
            n_categorical=cat_features.shape[1],
            n_classes=n_classes,
            class_names=class_names,
            class_distribution=class_distribution,
            is_imbalanced=is_imbalanced,
            has_high_dimensionality=features.shape[1] > 50,
            has_outliers=len(outlier_columns) > 0,
            has_strong_correlations=len(correlated_pairs) > 0,
            has_missing_values=has_missing,
            outlier_columns=outlier_columns,
            correlated_pairs=correlated_pairs,
            skewed_columns=skewed_columns,
            target_column=self.target_column,
            feature_columns=features.columns.tolist(),
            data_summary=data_summary,
        )
        return self.analysis

    # ──────────────────────────────────────────────
    # RECOMMEND (Gemini → heuristic fallback)
    # ──────────────────────────────────────────────
    def recommend(self, use_gemini: bool = True) -> list[ModelRecommendation]:
        if self.analysis is None:
            self.analyze()

        if use_gemini:
            try:
                recs = self._recommend_with_gemini()
                if recs and len(recs) >= 3:
                    return recs
            except Exception as e:
                print(f"[ModelRecommender] Gemini failed: {e}")

        print("[ModelRecommender] Using heuristic fallback")
        return self._recommend_heuristic()

    def _recommend_with_gemini(self) -> list[ModelRecommendation]:
        import requests
        
        # Get API key from environment
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key or api_key == "":
            raise ValueError("GEMINI_API_KEY not set in environment")

        is_clf = self.analysis.task_type != TaskType.REGRESSION
        model_type = "classification" if is_clf else "regression"
        available = {
            k: v["name"]
            for k, v in MODEL_CATALOG.items()
            if v["type"] == model_type
        }

        prompt = f"""You are an expert ML engineer. Analyze this dataset and recommend exactly 4 models ranked best to worst.

DATASET ANALYSIS:
{self.analysis.data_summary}

AVAILABLE MODELS (use ONLY these model_keys):
{json.dumps(available, indent=2)}

Respond with ONLY a JSON array of exactly 4 objects. No markdown, no backticks, no explanation.
Each object must have:
- "model_key": one of the keys above
- "reasoning": array of 2-3 specific reasons referencing actual data numbers
- "suggested_epochs": integer (50-200 for boosting, 100-500 for forests, 50 for linear)

IMPORTANT:
- Reference actual numbers from the analysis (row count, imbalance %, outliers, correlations).
- Do NOT recommend SVM or KNN if dataset > 10000 rows.
- 4 DIFFERENT models, no duplicates.
"""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048},
        }

        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        
        # Extract text from response
        text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        
        # Clean up JSON (remove markdown if present)
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        gemini_recs = json.loads(text.strip())

        if not isinstance(gemini_recs, list) or len(gemini_recs) < 3:
            raise ValueError(f"Bad response: {type(gemini_recs)}")

        recs, seen = [], set()
        for rank, rec in enumerate(gemini_recs[:4], 1):
            key = rec.get("model_key", "")
            if key not in MODEL_CATALOG or key in seen:
                continue
            seen.add(key)
            info = MODEL_CATALOG[key]
            is_pt = key in [
                "xgboost_clf", "xgboost_reg", "lightgbm_clf", "lightgbm_reg"
            ]
            recs.append(
                ModelRecommendation(
                    name=info["name"],
                    model_key=key,
                    rank=rank,
                    task_type=self.analysis.task_type,
                    reasoning=rec.get("reasoning", ["AI-selected"]),
                    strengths=info["strengths"],
                    weaknesses=info["weaknesses"],
                    is_pretrained=is_pt,
                    pretrained_note=(
                        "Supports warm-starting" if is_pt else ""
                    ),
                    estimated_train_time=info["time"],
                    complexity=info["complexity"],
                    suggested_epochs=rec.get("suggested_epochs", 50),
                    architecture_description=info["architecture"],
                )
            )

        for i, r in enumerate(recs, 1):
            r.rank = i
        return recs

    def _recommend_heuristic(self) -> list[ModelRecommendation]:
        a = self.analysis
        is_clf = a.task_type in [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        ]
        mtype = "classification" if is_clf else "regression"
        pool = {k: v for k, v in MODEL_CATALOG.items() if v["type"] == mtype}
        scored = []

        for key, info in pool.items():
            score, reasons = 0, []

            if a.n_samples < 1000:
                if key in [
                    "logistic_regression", "linear_regression",
                    "ridge_regression", "lasso_regression",
                    "naive_bayes_clf", "decision_tree_clf", "decision_tree_reg",
                ]:
                    score += 3
                    reasons.append(
                        f"Well-suited for small datasets ({a.n_samples:,} rows)"
                    )
                if "lightgbm" in key:
                    score -= 1
            elif a.n_samples > 10000:
                if key in ["svm_clf", "svr", "knn_clf"]:
                    score -= 3
                    reasons.append(f"May be slow on {a.n_samples:,} rows")
                if "lightgbm" in key:
                    score += 3
                    reasons.append(f"Optimized for large datasets ({a.n_samples:,} rows)")
                if "xgboost" in key:
                    score += 2
            else:
                if key in [
                    "random_forest_clf", "random_forest_reg",
                    "xgboost_clf", "xgboost_reg",
                ]:
                    score += 2
                    reasons.append("Good all-rounder for medium-sized data")

            if a.has_outliers and (
                "forest" in key or "xgboost" in key or "lightgbm" in key
            ):
                score += 1
                reasons.append("Robust to outliers in the data")

            if a.is_imbalanced and is_clf and key in [
                "xgboost_clf", "lightgbm_clf"
            ]:
                score += 2
                reasons.append("Handles class imbalance well")

            if a.has_strong_correlations and key in ["ridge_regression", "lasso_regression", "elastic_net"]:
                score += 2
                reasons.append("Handles multicollinearity via regularization")

            if a.n_categorical > 0 and "lightgbm" in key:
                score += 2
                reasons.append(
                    f"Native categorical support ({a.n_categorical} cat features)"
                )

            if key in [
                "xgboost_clf", "xgboost_reg",
                "lightgbm_clf", "lightgbm_reg",
            ]:
                score += 1
                reasons.append("Gradient boosting — top tabular performer")

            if key in ["logistic_regression", "linear_regression"]:
                score += 1
                reasons.append("Strong interpretable baseline")

            if not reasons:
                reasons.append("General-purpose model")

            scored.append((key, info, score, reasons))

        scored.sort(key=lambda x: x[2], reverse=True)

        recs = []
        for rank, (key, info, _, reasons) in enumerate(scored[:4], 1):
            is_pt = key in [
                "xgboost_clf", "xgboost_reg",
                "lightgbm_clf", "lightgbm_reg",
            ]
            se = 100 if ("forest" in key or "boost" in key or "gbm" in key) else 50
            recs.append(
                ModelRecommendation(
                    name=info["name"],
                    model_key=key,
                    rank=rank,
                    task_type=a.task_type,
                    reasoning=reasons,
                    strengths=info["strengths"],
                    weaknesses=info["weaknesses"],
                    is_pretrained=is_pt,
                    pretrained_note=(
                        "Supports incremental learning" if is_pt else ""
                    ),
                    estimated_train_time=info["time"],
                    complexity=info["complexity"],
                    suggested_epochs=se,
                    architecture_description=info["architecture"],
                )
            )
        return recs