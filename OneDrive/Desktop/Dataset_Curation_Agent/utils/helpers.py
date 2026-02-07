"""Utility functions used across modules."""

import pandas as pd
import streamlit as st
from io import BytesIO


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