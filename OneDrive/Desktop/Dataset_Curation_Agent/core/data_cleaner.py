"""Automated data cleaning pipeline."""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from config.settings import NULL_DROP_THRESHOLD, CATEGORICAL_THRESHOLD


@dataclass
class CleaningReport:
    """Stores a detailed report of all cleaning actions performed."""
    original_shape: tuple = (0, 0)
    cleaned_shape: tuple = (0, 0)
    nulls_before: dict = field(default_factory=dict)
    nulls_after: dict = field(default_factory=dict)
    total_nulls_before: int = 0
    total_nulls_after: int = 0
    duplicates_removed: int = 0
    columns_dropped: list = field(default_factory=list)
    columns_type_converted: dict = field(default_factory=dict)
    whitespace_cleaned: list = field(default_factory=list)
    actions_log: list = field(default_factory=list)


class DataCleaner:
    """
    Automated data cleaning pipeline.
    
    Steps:
        1. Remove fully empty rows/columns
        2. Strip whitespace from string columns
        3. Drop high-null columns (>threshold)
        4. Fill remaining nulls (median for numeric, mode for categorical)
        5. Remove duplicate rows
        6. Optimize dtypes (downcast numerics, convert low-cardinality to category)
    """

    def __init__(self, null_threshold: float = NULL_DROP_THRESHOLD):
        self.null_threshold = null_threshold
        self.report = CleaningReport()

    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
        """
        Run the full cleaning pipeline.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Tuple of (cleaned DataFrame, CleaningReport)
        """
        self.report = CleaningReport()
        self.report.original_shape = df.shape
        self.report.nulls_before = df.isnull().sum().to_dict()
        self.report.total_nulls_before = int(df.isnull().sum().sum())

        df = df.copy()

        df = self._remove_empty_rows_cols(df)
        df = self._clean_whitespace(df)
        df = self._drop_high_null_columns(df)
        df = self._fill_nulls(df)
        df = self._remove_duplicates(df)
        df = self._optimize_dtypes(df)

        self.report.cleaned_shape = df.shape
        self.report.nulls_after = df.isnull().sum().to_dict()
        self.report.total_nulls_after = int(df.isnull().sum().sum())

        return df, self.report

    def _remove_empty_rows_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove completely empty rows and columns."""
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            df = df.drop(columns=empty_cols)
            self.report.actions_log.append(
                f"Dropped {len(empty_cols)} fully empty column(s): {empty_cols}"
            )

        empty_rows_count = df.isnull().all(axis=1).sum()
        if empty_rows_count > 0:
            df = df.dropna(how="all")
            self.report.actions_log.append(
                f"Dropped {empty_rows_count} fully empty row(s)"
            )

        return df

    def _clean_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip leading/trailing whitespace from string columns."""
        str_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip().replace("nan", np.nan)

        if str_cols:
            self.report.whitespace_cleaned = str_cols
            self.report.actions_log.append(
                f"Stripped whitespace from {len(str_cols)} text column(s)"
            )
        return df

    def _drop_high_null_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns where null percentage exceeds threshold."""
        null_pct = df.isnull().mean()
        high_null_cols = null_pct[null_pct > self.null_threshold].index.tolist()

        if high_null_cols:
            df = df.drop(columns=high_null_cols)
            self.report.columns_dropped = high_null_cols
            self.report.actions_log.append(
                f"Dropped {len(high_null_cols)} column(s) with >{self.null_threshold*100:.0f}% nulls: {high_null_cols}"
            )
        return df

    def _fill_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill remaining nulls: median for numeric, mode for categorical."""
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                self.report.actions_log.append(
                    f"Filled nulls in '{col}' with median ({median_val:.2f})"
                )

        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
                    self.report.actions_log.append(
                        f"Filled nulls in '{col}' with mode ('{mode_val[0]}')"
                    )
                else:
                    df[col] = df[col].fillna("Unknown")
                    self.report.actions_log.append(
                        f"Filled nulls in '{col}' with 'Unknown'"
                    )
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            df = df.drop_duplicates().reset_index(drop=True)
            self.report.duplicates_removed = int(dup_count)
            self.report.actions_log.append(
                f"Removed {dup_count} duplicate row(s)"
            )
        return df

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downcast numeric types and convert low-cardinality strings to category."""
        conversions = {}

        # Downcast integers
        int_cols = df.select_dtypes(include=["int64"]).columns
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], downcast="integer")
            if df[col].dtype != np.int64:
                conversions[col] = f"int64 → {df[col].dtype}"

        # Downcast floats
        float_cols = df.select_dtypes(include=["float64"]).columns
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast="float")
            if df[col].dtype != np.float64:
                conversions[col] = f"float64 → {df[col].dtype}"

        # Low-cardinality strings → category
        obj_cols = df.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            if df[col].nunique() < CATEGORICAL_THRESHOLD:
                df[col] = df[col].astype("category")
                conversions[col] = "object → category"

        if conversions:
            self.report.columns_type_converted = conversions
            self.report.actions_log.append(
                f"Optimized dtypes for {len(conversions)} column(s)"
            )
        return df