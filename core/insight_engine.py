"""
Insight Engine — Automatically detects patterns, anomalies, and relationships.

This is the analytical brain of Sprint 2. It scans the cleaned DataFrame
and produces structured Insight objects that the StoryGenerator turns into narrative.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from config.settings import (
    OUTLIER_IQR_MULTIPLIER,
    CORRELATION_STRONG_THRESHOLD,
    CORRELATION_MODERATE_THRESHOLD,
    SKEW_THRESHOLD,
    IMBALANCE_RATIO_THRESHOLD,
    HIGH_CARDINALITY_THRESHOLD,
    LOW_VARIANCE_THRESHOLD,
)


class InsightType(Enum):
    OUTLIER = "outlier"
    CORRELATION = "correlation"
    SKEWNESS = "skewness"
    IMBALANCE = "imbalance"
    HIGH_CARDINALITY = "high_cardinality"
    LOW_VARIANCE = "low_variance"
    MISSING_PATTERN = "missing_pattern"
    DISTRIBUTION = "distribution"
    TREND = "trend"
    SUMMARY = "summary"


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Insight:
    """A single detected insight about the data."""
    insight_type: InsightType
    severity: Severity
    title: str
    description: str
    affected_columns: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    recommendation: str = ""


class InsightEngine:
    """
    Scans a DataFrame and produces a list of Insight objects.
    
    Detection pipeline:
        1. Dataset summary
        2. Outlier detection (IQR method)
        3. Correlation analysis
        4. Skewness detection
        5. Class imbalance detection
        6. High cardinality detection
        7. Low variance detection
        8. Distribution analysis
    """

    def __init__(self, df: pd.DataFrame, raw_df: pd.DataFrame = None):
        self.df = df
        self.raw_df = raw_df if raw_df is not None else df
        self.insights: list[Insight] = []

    def run(self) -> list[Insight]:
        """Execute all detection routines and return insights."""
        self.insights = []

        self._detect_dataset_summary()
        self._detect_outliers()
        self._detect_correlations()
        self._detect_skewness()
        self._detect_class_imbalance()
        self._detect_high_cardinality()
        self._detect_low_variance()
        self._detect_distribution_types()

        # Sort by severity: critical first, then warning, then info
        severity_order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
        self.insights.sort(key=lambda x: severity_order[x.severity])

        return self.insights

    def _detect_dataset_summary(self):
        """Generate high-level dataset summary insight."""
        n_rows, n_cols = self.df.shape
        numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()

        self.insights.append(Insight(
            insight_type=InsightType.SUMMARY,
            severity=Severity.INFO,
            title="Dataset Overview",
            description=(
                f"The dataset contains {n_rows:,} rows and {n_cols} columns. "
                f"There are {len(numeric_cols)} numeric feature(s) and "
                f"{len(cat_cols)} categorical feature(s)."
            ),
            metrics={
                "rows": n_rows,
                "cols": n_cols,
                "numeric_cols": len(numeric_cols),
                "categorical_cols": len(cat_cols),
            },
        ))

    def _detect_outliers(self):
        """Detect outliers using the IQR method for each numeric column."""
        numeric_df = self.df.select_dtypes(include=["number"])

        outlier_summary = {}
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if series.empty:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
            upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr
            outliers = series[(series < lower) | (series > upper)]

            if len(outliers) > 0:
                pct = len(outliers) / len(series) * 100
                outlier_summary[col] = {
                    "count": len(outliers),
                    "percentage": round(pct, 2),
                    "lower_bound": round(lower, 2),
                    "upper_bound": round(upper, 2),
                    "min_outlier": round(outliers.min(), 2),
                    "max_outlier": round(outliers.max(), 2),
                }

        if outlier_summary:
            # Determine severity
            max_pct = max(v["percentage"] for v in outlier_summary.values())
            severity = (
                Severity.CRITICAL if max_pct > 10
                else Severity.WARNING if max_pct > 5
                else Severity.INFO
            )

            cols_with_outliers = list(outlier_summary.keys())
            top_offenders = sorted(
                outlier_summary.items(), key=lambda x: x[1]["percentage"], reverse=True
            )[:3]

            desc_parts = [f"Detected outliers in {len(cols_with_outliers)} column(s). "]
            for col_name, stats in top_offenders:
                desc_parts.append(
                    f"'{col_name}' has {stats['count']} outlier(s) ({stats['percentage']}% of data, "
                    f"range: {stats['min_outlier']} to {stats['max_outlier']}). "
                )

            self.insights.append(Insight(
                insight_type=InsightType.OUTLIER,
                severity=severity,
                title=f"Outliers Detected in {len(cols_with_outliers)} Column(s)",
                description="".join(desc_parts),
                affected_columns=cols_with_outliers,
                metrics=outlier_summary,
                recommendation=(
                    "Consider investigating these outliers. They may represent data entry errors, "
                    "genuine extreme values, or domain-specific edge cases. Options include capping, "
                    "removing, or applying robust scaling for ML pipelines."
                ),
            ))

    def _detect_correlations(self):
        """Find strongly and moderately correlated feature pairs."""
        numeric_df = self.df.select_dtypes(include=["number"])
        if numeric_df.shape[1] < 2:
            return

        corr_matrix = numeric_df.corr().abs()

        strong_pairs = []
        moderate_pairs = []

        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr_matrix.iloc[i, j]
                pair = (cols[i], cols[j], round(val, 3))

                if val >= CORRELATION_STRONG_THRESHOLD:
                    strong_pairs.append(pair)
                elif val >= CORRELATION_MODERATE_THRESHOLD:
                    moderate_pairs.append(pair)

        if strong_pairs:
            strong_pairs.sort(key=lambda x: x[2], reverse=True)
            desc_parts = [f"Found {len(strong_pairs)} strongly correlated pair(s) (|r| ≥ {CORRELATION_STRONG_THRESHOLD}): "]
            for c1, c2, r in strong_pairs[:5]:
                desc_parts.append(f"'{c1}' ↔ '{c2}' (r={r}). ")

            self.insights.append(Insight(
                insight_type=InsightType.CORRELATION,
                severity=Severity.WARNING,
                title=f"{len(strong_pairs)} Strongly Correlated Feature Pair(s)",
                description="".join(desc_parts),
                affected_columns=list({c for p in strong_pairs for c in p[:2]}),
                metrics={"strong_pairs": [(c1, c2, r) for c1, c2, r in strong_pairs]},
                recommendation=(
                    "Highly correlated features may cause multicollinearity in linear models. "
                    "Consider dropping one from each pair or using PCA/feature selection."
                ),
            ))

        if moderate_pairs:
            self.insights.append(Insight(
                insight_type=InsightType.CORRELATION,
                severity=Severity.INFO,
                title=f"{len(moderate_pairs)} Moderately Correlated Pair(s)",
                description=f"Found {len(moderate_pairs)} moderately correlated feature pair(s) (|r| between {CORRELATION_MODERATE_THRESHOLD} and {CORRELATION_STRONG_THRESHOLD}).",
                affected_columns=list({c for p in moderate_pairs for c in p[:2]}),
                metrics={"moderate_pairs": [(c1, c2, r) for c1, c2, r in moderate_pairs]},
            ))

    def _detect_skewness(self):
        """Detect highly skewed numeric distributions."""
        numeric_df = self.df.select_dtypes(include=["number"])
        skewed_cols = {}

        for col in numeric_df.columns:
            skew_val = numeric_df[col].skew()
            if abs(skew_val) > SKEW_THRESHOLD:
                skewed_cols[col] = round(skew_val, 3)

        if skewed_cols:
            severity = (
                Severity.WARNING
                if any(abs(v) > 2 for v in skewed_cols.values())
                else Severity.INFO
            )

            desc_parts = [f"{len(skewed_cols)} column(s) show significant skewness (|skew| > {SKEW_THRESHOLD}): "]
            for col_name, skew_val in sorted(skewed_cols.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
                direction = "right (positive)" if skew_val > 0 else "left (negative)"
                desc_parts.append(f"'{col_name}' is skewed {direction} (skew={skew_val}). ")

            self.insights.append(Insight(
                insight_type=InsightType.SKEWNESS,
                severity=severity,
                title=f"{len(skewed_cols)} Skewed Distribution(s) Detected",
                description="".join(desc_parts),
                affected_columns=list(skewed_cols.keys()),
                metrics=skewed_cols,
                recommendation=(
                    "Skewed features can hurt model performance. Consider log, square-root, "
                    "or Box-Cox transformations to normalize these distributions."
                ),
            ))

    def _detect_class_imbalance(self):
        """Detect class imbalance in categorical columns."""
        cat_df = self.df.select_dtypes(include=["object", "category"])
        imbalanced = {}

        for col in cat_df.columns:
            counts = cat_df[col].value_counts(normalize=True)
            if counts.empty:
                continue

            top_pct = counts.iloc[0]
            if top_pct > IMBALANCE_RATIO_THRESHOLD:
                imbalanced[col] = {
                    "dominant_class": str(counts.index[0]),
                    "dominant_pct": round(top_pct * 100, 2),
                    "n_classes": len(counts),
                    "minority_class": str(counts.index[-1]),
                    "minority_pct": round(counts.iloc[-1] * 100, 2),
                }

        if imbalanced:
            severity = (
                Severity.CRITICAL
                if any(v["dominant_pct"] > 90 for v in imbalanced.values())
                else Severity.WARNING
            )

            desc_parts = [f"{len(imbalanced)} categorical column(s) show class imbalance: "]
            for col_name, info in imbalanced.items():
                desc_parts.append(
                    f"'{col_name}' is dominated by '{info['dominant_class']}' "
                    f"at {info['dominant_pct']}% (minority: '{info['minority_class']}' at {info['minority_pct']}%). "
                )

            self.insights.append(Insight(
                insight_type=InsightType.IMBALANCE,
                severity=severity,
                title=f"Class Imbalance in {len(imbalanced)} Column(s)",
                description="".join(desc_parts),
                affected_columns=list(imbalanced.keys()),
                metrics=imbalanced,
                recommendation=(
                    "If these columns are prediction targets, consider SMOTE, undersampling, "
                    "or class-weighted models to address the imbalance."
                ),
            ))

    def _detect_high_cardinality(self):
        """Detect categorical columns with too many unique values."""
        cat_df = self.df.select_dtypes(include=["object", "category"])
        high_card = {}

        for col in cat_df.columns:
            nunique = cat_df[col].nunique()
            if nunique > HIGH_CARDINALITY_THRESHOLD:
                high_card[col] = nunique

        if high_card:
            desc = ", ".join(f"'{c}' ({n} unique)" for c, n in high_card.items())
            self.insights.append(Insight(
                insight_type=InsightType.HIGH_CARDINALITY,
                severity=Severity.WARNING,
                title=f"High Cardinality in {len(high_card)} Column(s)",
                description=f"Columns with unusually many unique values: {desc}.",
                affected_columns=list(high_card.keys()),
                metrics=high_card,
                recommendation=(
                    "High-cardinality categoricals are hard to encode for ML. Consider frequency "
                    "encoding, target encoding, or grouping rare categories into an 'Other' bucket."
                ),
            ))

    def _detect_low_variance(self):
        """Detect numeric columns with near-zero variance."""
        numeric_df = self.df.select_dtypes(include=["number"])
        low_var = {}

        for col in numeric_df.columns:
            var = numeric_df[col].var()
            if var < LOW_VARIANCE_THRESHOLD:
                low_var[col] = round(var, 6)

        if low_var:
            desc = ", ".join(f"'{c}' (var={v})" for c, v in low_var.items())
            self.insights.append(Insight(
                insight_type=InsightType.LOW_VARIANCE,
                severity=Severity.WARNING,
                title=f"Near-Zero Variance in {len(low_var)} Column(s)",
                description=f"These columns carry almost no information: {desc}.",
                affected_columns=list(low_var.keys()),
                metrics=low_var,
                recommendation=(
                    "Low-variance features add noise and no predictive power. "
                    "Consider removing them before model training."
                ),
            ))

    def _detect_distribution_types(self):
        """Classify distributions of numeric columns."""
        numeric_df = self.df.select_dtypes(include=["number"])
        dist_info = {}

        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if series.empty:
                continue

            skew = series.skew()
            kurt = series.kurtosis()
            nunique = series.nunique()
            n = len(series)

            # Simple heuristic classification
            if nunique <= 2:
                dist_type = "Binary"
            elif nunique <= 10 and nunique / n < 0.05:
                dist_type = "Discrete"
            elif abs(skew) < 0.5 and abs(kurt) < 1:
                dist_type = "Near-Normal"
            elif skew > SKEW_THRESHOLD:
                dist_type = "Right-Skewed"
            elif skew < -SKEW_THRESHOLD:
                dist_type = "Left-Skewed"
            elif kurt > 3:
                dist_type = "Heavy-Tailed"
            else:
                dist_type = "Mixed"

            dist_info[col] = {
                "type": dist_type,
                "skew": round(skew, 3),
                "kurtosis": round(kurt, 3),
            }

        if dist_info:
            # Group by type
            type_groups = {}
            for col_name, info in dist_info.items():
                t = info["type"]
                if t not in type_groups:
                    type_groups[t] = []
                type_groups[t].append(col_name)

            desc_parts = ["Distribution classification: "]
            for t, cols in type_groups.items():
                desc_parts.append(f"{t}: {', '.join(cols)}. ")

            self.insights.append(Insight(
                insight_type=InsightType.DISTRIBUTION,
                severity=Severity.INFO,
                title="Distribution Analysis",
                description="".join(desc_parts),
                metrics=dist_info,
            ))