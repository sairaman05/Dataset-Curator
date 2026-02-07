"""Generate data summaries, statistics, and dtype profiles."""

import pandas as pd
import numpy as np
from core.data_cleaner import CleaningReport


class DataProfiler:
    """Profile a DataFrame — shape, types, stats, null info."""

    def __init__(self, raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, report: CleaningReport):
        self.raw_df = raw_df
        self.cleaned_df = cleaned_df
        self.report = report

    def get_overview(self) -> dict:
        """High-level overview of the dataset."""
        return {
            "rows_before": self.report.original_shape[0],
            "cols_before": self.report.original_shape[1],
            "rows_after": self.report.cleaned_shape[0],
            "cols_after": self.report.cleaned_shape[1],
            "rows_removed": self.report.original_shape[0] - self.report.cleaned_shape[0],
            "cols_removed": self.report.original_shape[1] - self.report.cleaned_shape[1],
            "total_nulls_before": self.report.total_nulls_before,
            "total_nulls_after": self.report.total_nulls_after,
            "nulls_resolved": self.report.total_nulls_before - self.report.total_nulls_after,
            "duplicates_removed": self.report.duplicates_removed,
        }

    def get_column_info(self) -> pd.DataFrame:
        """Detailed per-column information table."""
        info_data = []
        for col in self.cleaned_df.columns:
            series = self.cleaned_df[col]
            null_before = self.report.nulls_before.get(col, 0)
            null_after = self.report.nulls_after.get(col, 0)

            info_data.append({
                "Column": col,
                "Dtype": str(series.dtype),
                "Non-Null Count": int(series.notnull().sum()),
                "Null Before": int(null_before),
                "Null After": int(null_after),
                "Unique Values": int(series.nunique()),
                "Sample Value": str(series.dropna().iloc[0]) if not series.dropna().empty else "N/A",
            })

        return pd.DataFrame(info_data)

    def get_numeric_stats(self) -> pd.DataFrame:
        """Descriptive statistics for numeric columns."""
        numeric_df = self.cleaned_df.select_dtypes(include=["number"])
        if numeric_df.empty:
            return pd.DataFrame()

        stats = numeric_df.describe().T
        stats["skew"] = numeric_df.skew()
        stats["kurtosis"] = numeric_df.kurtosis()
        stats = stats.round(3)
        return stats

    def get_categorical_stats(self) -> pd.DataFrame:
        """Stats for categorical/object columns."""
        cat_df = self.cleaned_df.select_dtypes(include=["object", "category"])
        if cat_df.empty:
            return pd.DataFrame()

        cat_stats = []
        for col in cat_df.columns:
            series = cat_df[col]
            top_val = series.mode()[0] if not series.mode().empty else "N/A"
            cat_stats.append({
                "Column": col,
                "Unique Values": series.nunique(),
                "Most Frequent": top_val,
                "Frequency": int((series == top_val).sum()),
                "Frequency %": round((series == top_val).mean() * 100, 2),
            })

        return pd.DataFrame(cat_stats)

    def get_null_comparison(self) -> pd.DataFrame:
        """Before vs after null comparison per column."""
        cols = list(self.report.nulls_before.keys())
        comparison = []
        for col in cols:
            before = self.report.nulls_before.get(col, 0)
            after = self.report.nulls_after.get(col, 0)
            comparison.append({
                "Column": col,
                "Nulls Before": int(before),
                "Nulls After": int(after),
                "Nulls Resolved": int(before - after),
            })
        return pd.DataFrame(comparison)