# agents/fe_agent.py
"""
Feature Engineering Agent — orchestrates all transforms with undo/redo.

5 Categories:
  1. Creation      — Polynomial, Arithmetic, Aggregation
  2. Transformation — Encoding, Math transforms (log, sqrt, boxcox, binning)
  3. Extraction    — PCA, DateTime extraction
  4. Selection     — Filter, Wrapper, Embedded, Auto
  5. Scaling       — Standard, MinMax, Robust, MaxAbs
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from core.feature_engineer import (
    FELog, FEResult,
    # 1. Creation
    create_polynomial, create_arithmetic, create_aggregation,
    # 2. Transformation
    encode_label, encode_onehot, encode_target, encode_frequency,
    transform_log, transform_sqrt, transform_boxcox, transform_binning,
    # 3. Extraction
    extract_pca, extract_datetime,
    # 4. Selection
    select_variance, select_correlation, select_mutual_info, select_statistical,
    select_rfe, select_lasso, select_tree_importance, auto_select_features,
    # 5. Scaling
    scale_standard, scale_minmax, scale_robust, scale_maxabs,
    # Utility
    drop_columns,
)


class FEAgent:
    """Manages the feature engineering pipeline with undo support."""

    def __init__(self, original_df: pd.DataFrame):
        self.original_df = original_df.copy()
        self.current_df = original_df.copy()
        self.logs: List[FELog] = []
        self._snapshots: List[pd.DataFrame] = [original_df.copy()]

    # ─── Properties ──────────────────────────────────────

    @property
    def shape(self):
        return self.current_df.shape

    @property
    def columns(self):
        return list(self.current_df.columns)

    @property
    def numeric_columns(self):
        return list(self.current_df.select_dtypes(include=[np.number]).columns)

    @property
    def categorical_columns(self):
        return list(self.current_df.select_dtypes(include=["object", "category"]).columns)

    @property
    def datetime_columns(self):
        cols = []
        for col in self.current_df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.current_df[col]):
                cols.append(col)
                continue
            try:
                sample = self.current_df[col].dropna().head(20)
                if len(sample) > 0:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().sum() > len(sample) * 0.7:
                        cols.append(col)
            except Exception:
                pass
        return cols

    # ─── Internal Apply ──────────────────────────────────

    def _apply(self, func, *args, **kwargs) -> Tuple[FELog, Any]:
        """Apply transform, save snapshot, return (log, extras)."""
        result = func(self.current_df, *args, **kwargs)

        if isinstance(result, tuple) and len(result) == 3:
            df, extra, log = result
        else:
            df, log = result
            extra = None

        self._snapshots.append(self.current_df.copy())
        self.current_df = df
        self.logs.append(log)
        return log, extra

    # ═══════════════════════════════════════════════════════
    # 1. FEATURE CREATION
    # ═══════════════════════════════════════════════════════

    def apply_polynomial(self, columns, degree=2, interaction_only=False):
        return self._apply(create_polynomial, columns, degree=degree, interaction_only=interaction_only)

    def apply_arithmetic(self, col_a, col_b, operations=None):
        return self._apply(create_arithmetic, col_a, col_b, operations=operations)

    def apply_aggregation(self, columns, agg_funcs=None):
        return self._apply(create_aggregation, columns, agg_funcs=agg_funcs)

    # ═══════════════════════════════════════════════════════
    # 2. FEATURE TRANSFORMATION
    # ═══════════════════════════════════════════════════════

    def apply_label_encoding(self, columns):
        return self._apply(encode_label, columns)

    def apply_onehot_encoding(self, columns, drop_first=True, max_categories=20):
        return self._apply(encode_onehot, columns, drop_first=drop_first, max_categories=max_categories)

    def apply_target_encoding(self, columns, target_col):
        return self._apply(encode_target, columns, target_col=target_col)

    def apply_frequency_encoding(self, columns):
        return self._apply(encode_frequency, columns)

    def apply_log_transform(self, columns):
        return self._apply(transform_log, columns)

    def apply_sqrt_transform(self, columns):
        return self._apply(transform_sqrt, columns)

    def apply_boxcox_transform(self, columns):
        return self._apply(transform_boxcox, columns)

    def apply_binning(self, columns, n_bins=5, strategy="quantile"):
        return self._apply(transform_binning, columns, n_bins=n_bins, strategy=strategy)

    # ═══════════════════════════════════════════════════════
    # 3. FEATURE EXTRACTION
    # ═══════════════════════════════════════════════════════

    def apply_pca(self, columns, n_components=5):
        return self._apply(extract_pca, columns, n_components=n_components)

    def apply_datetime_extraction(self, columns, cyclical=True):
        return self._apply(extract_datetime, columns, cyclical=cyclical)

    # ═══════════════════════════════════════════════════════
    # 4. FEATURE SELECTION
    # ═══════════════════════════════════════════════════════

    # Filter
    def apply_variance_threshold(self, columns, threshold=0.01):
        return self._apply(select_variance, columns, threshold=threshold)

    def apply_correlation_filter(self, columns, threshold=0.95):
        return self._apply(select_correlation, columns, threshold=threshold)

    def apply_mutual_info(self, feature_cols, target_col, task_type, k=10):
        return self._apply(select_mutual_info, feature_cols, target_col=target_col, task_type=task_type, k=k)

    def apply_statistical_test(self, feature_cols, target_col, task_type, k=10):
        return self._apply(select_statistical, feature_cols, target_col=target_col, task_type=task_type, k=k)

    # Wrapper
    def apply_rfe(self, feature_cols, target_col, task_type, n_features=10):
        return self._apply(select_rfe, feature_cols, target_col=target_col, task_type=task_type, n_features=n_features)

    # Embedded
    def apply_lasso_selection(self, feature_cols, target_col, task_type, alpha=0.01):
        return self._apply(select_lasso, feature_cols, target_col=target_col, task_type=task_type, alpha=alpha)

    def apply_tree_importance(self, feature_cols, target_col, task_type, threshold=0.01):
        return self._apply(select_tree_importance, feature_cols, target_col=target_col, task_type=task_type, threshold=threshold)

    # Auto
    def apply_auto_select(self, feature_cols, target_col, task_type, top_k=10):
        return self._apply(auto_select_features, feature_cols, target_col=target_col, task_type=task_type, top_k=top_k)

    # ═══════════════════════════════════════════════════════
    # 5. FEATURE SCALING
    # ═══════════════════════════════════════════════════════

    def apply_standard_scaling(self, columns):
        return self._apply(scale_standard, columns)

    def apply_minmax_scaling(self, columns):
        return self._apply(scale_minmax, columns)

    def apply_robust_scaling(self, columns):
        return self._apply(scale_robust, columns)

    def apply_maxabs_scaling(self, columns):
        return self._apply(scale_maxabs, columns)

    # ═══════════════════════════════════════════════════════
    # UTILITY
    # ═══════════════════════════════════════════════════════

    def apply_drop_columns(self, columns):
        return self._apply(drop_columns, columns)

    def undo(self) -> bool:
        if len(self._snapshots) > 1:
            self.current_df = self._snapshots.pop()
            if self.logs:
                self.logs.pop()
            return True
        return False

    def reset(self):
        self.current_df = self.original_df.copy()
        self.logs.clear()
        self._snapshots = [self.original_df.copy()]

    def get_result(self) -> FEResult:
        orig = set(self.original_df.columns)
        curr = set(self.current_df.columns)
        return FEResult(
            original_df=self.original_df,
            engineered_df=self.current_df,
            logs=self.logs,
            summary={
                "features_added": len(curr - orig),
                "features_removed": len(orig - curr),
                "original_columns": len(orig),
                "current_columns": len(curr),
                "transforms_applied": len(self.logs),
            },
        )

    def get_comparison_stats(self) -> Dict:
        o, c = self.original_df, self.current_df
        return {
            "rows": {"before": len(o), "after": len(c)},
            "columns": {"before": len(o.columns), "after": len(c.columns)},
            "numeric": {
                "before": len(o.select_dtypes(include=[np.number]).columns),
                "after": len(c.select_dtypes(include=[np.number]).columns),
            },
            "categorical": {
                "before": len(o.select_dtypes(include=["object", "category"]).columns),
                "after": len(c.select_dtypes(include=["object", "category"]).columns),
            },
            "memory_mb": {
                "before": round(o.memory_usage(deep=True).sum() / 1024 ** 2, 2),
                "after": round(c.memory_usage(deep=True).sum() / 1024 ** 2, 2),
            },
        }