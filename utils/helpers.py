"""Utility functions used across modules."""

import pandas as pd
import numpy as np
from io import BytesIO
from typing import Tuple

try:
    import streamlit as st
except ImportError:
    pass


# ═══════════════════════════════════════════════════════
# Existing helpers (Sprint 1 / Sprint 2)
# ═══════════════════════════════════════════════════════

def format_number(n: int) -> str:
    """Format large numbers with commas."""
    return f"{n:,}"


def format_percentage(value: float) -> str:
    """Format float as percentage string."""
    return f"{value:.2f}%"


def get_memory_usage(df: pd.DataFrame) -> str:
    """Get human-readable memory usage of a DataFrame."""
    bytes_used = df.memory_usage(deep=True).sum()
    if bytes_used < 1024:
        return f"{bytes_used} B"
    elif bytes_used < 1024 ** 2:
        return f"{bytes_used / 1024:.2f} KB"
    elif bytes_used < 1024 ** 3:
        return f"{bytes_used / (1024 ** 2):.2f} MB"
    else:
        return f"{bytes_used / (1024 ** 3):.2f} GB"


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to Excel bytes for download."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Cleaned Data")
    return buffer.getvalue()


# ═══════════════════════════════════════════════════════
# ML helpers (Sprint 3)
# ═══════════════════════════════════════════════════════

def safe_to_float(series: pd.Series) -> np.ndarray:
    """Convert a pandas Series to float64 numpy array using multiple strategies."""

    # Strategy 1: Direct cast
    try:
        arr = np.array(series, dtype=np.float64)
        if not np.all(np.isnan(arr)):
            return arr
    except (ValueError, TypeError):
        pass

    # Strategy 2: pd.to_numeric
    try:
        numeric = pd.to_numeric(series, errors="coerce")
        arr = numeric.to_numpy(dtype=np.float64)
        if np.sum(~np.isnan(arr)) > 0:
            nan_mask = np.isnan(arr)
            if nan_mask.any():
                arr[nan_mask] = np.nanmedian(arr[~nan_mask]) if np.any(~nan_mask) else 0.0
            return arr
    except Exception:
        pass

    # Strategy 3: Datetime
    try:
        if pd.api.types.is_datetime64_any_dtype(series):
            arr = series.values.view("int64").astype(np.float64) / 1e9
            return arr
    except Exception:
        pass

    # Strategy 4: String datetime parse
    try:
        dt = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
        if dt.notna().sum() > len(series) * 0.5:
            arr = dt.values.view("int64").astype(np.float64) / 1e9
            nan_mask = np.isnan(arr)
            if nan_mask.any():
                arr[nan_mask] = np.nanmedian(arr[~nan_mask]) if np.any(~nan_mask) else 0.0
            return arr
    except Exception:
        pass

    # Strategy 5: Label encode
    try:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        non_null = series.fillna("__MISSING__").astype(str)
        arr = le.fit_transform(non_null).astype(np.float64)
        return arr
    except Exception:
        pass

    # Strategy 6: Hash fallback
    arr = np.array([abs(hash(str(v))) % (10**8) for v in series], dtype=np.float64)
    return arr


def nuke_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove or convert all datetime columns to numeric."""
    df = df.copy()
    dt_cols = []

    for col in df.columns:
        is_dt = False
        try:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                is_dt = True
        except Exception:
            pass
        if not is_dt:
            try:
                if "datetime" in str(df[col].dtype).lower():
                    is_dt = True
            except Exception:
                pass
        if not is_dt:
            try:
                vals = df[col].values
                if "datetime" in str(vals.dtype).lower():
                    is_dt = True
            except Exception:
                pass
        if not is_dt:
            try:
                first = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                if isinstance(first, (pd.Timestamp, np.datetime64)):
                    is_dt = True
            except Exception:
                pass
        if is_dt:
            dt_cols.append(col)

    for col in dt_cols:
        try:
            ts = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_year"] = ts.dt.year.astype(float)
            df[f"{col}_month"] = ts.dt.month.astype(float)
            df[f"{col}_day"] = ts.dt.day.astype(float)
            df[f"{col}_dow"] = ts.dt.dayofweek.astype(float)
            df[f"{col}_epoch"] = ts.values.view("int64").astype(np.float64) / 1e9
            df.drop(columns=[col], inplace=True)
        except Exception:
            df.drop(columns=[col], inplace=True)

    return df


def detect_task_type(series: pd.Series) -> Tuple[str, int]:
    """Detect if target is classification or regression.
    Returns: (task_type, n_classes)
    """
    from config.settings import CLASSIFICATION_UNIQUE_THRESHOLD

    nunique = series.nunique()

    # Bool always classification
    if pd.api.types.is_bool_dtype(series):
        return ("binary_classification", 2)

    # Check if this is a string/categorical type
    # Covers: object, category, StringDtype (pandas 2.x), and ArrowDtype[string]
    def _is_string_like(s):
        if pd.api.types.is_object_dtype(s):
            return True
        try:
            if pd.api.types.is_categorical_dtype(s):
                return True
        except Exception:
            pass
        if pd.api.types.is_string_dtype(s) and not pd.api.types.is_numeric_dtype(s):
            return True
        dtype_str = str(s.dtype).lower()
        if "string" in dtype_str or "category" in dtype_str or "object" in dtype_str:
            return True
        return False

    if _is_string_like(series):
        if nunique > CLASSIFICATION_UNIQUE_THRESHOLD:
            # Check if almost ALL values are purely numeric (e.g., "100.0", "250.0")
            # This catches numeric columns that got cast to category/object by the cleaner
            sample = series.dropna()
            if len(sample) > 0:
                check_vals = sample.astype(str).head(min(200, len(sample)))
                numeric_count = 0
                for v in check_vals:
                    try:
                        float(v)
                        numeric_count += 1
                    except (ValueError, TypeError):
                        pass
                pct_numeric = numeric_count / len(check_vals)
                if pct_numeric > 0.95:
                    return ("regression", nunique)
        # Truly categorical
        return ("binary_classification" if nunique == 2 else "multiclass_classification", nunique)

    # Numeric dtype with few unique values → classification
    if nunique <= CLASSIFICATION_UNIQUE_THRESHOLD:
        return ("binary_classification" if nunique == 2 else "multiclass_classification", nunique)

    # Numeric with many unique values → regression
    return ("regression", nunique)