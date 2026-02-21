"""Handles file upload parsing for various formats."""

import pandas as pd
import streamlit as st
from config.settings import SUPPORTED_FILE_TYPES


class DataLoader:
    """Load data from uploaded files into pandas DataFrames."""

    PARSERS = {
        "csv": "_read_csv",
        "tsv": "_read_tsv",
        "xlsx": "_read_excel",
        "xls": "_read_excel",
        "json": "_read_json",
        "parquet": "_read_parquet",
    }

    def load(self, uploaded_file) -> pd.DataFrame:
        """
        Parse an uploaded file and return a DataFrame.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            pd.DataFrame with raw data
            
        Raises:
            ValueError: If file type is unsupported
        """
        file_ext = uploaded_file.name.split(".")[-1].lower()

        if file_ext not in self.PARSERS:
            raise ValueError(
                f"Unsupported file type: .{file_ext}. "
                f"Supported: {', '.join(SUPPORTED_FILE_TYPES)}"
            )

        parser_method = getattr(self, self.PARSERS[file_ext])
        df = parser_method(uploaded_file)
        return df

    def _read_csv(self, file) -> pd.DataFrame:
        """Read CSV with automatic delimiter detection."""
        try:
            return pd.read_csv(file, encoding="utf-8")
        except UnicodeDecodeError:
            file.seek(0)
            return pd.read_csv(file, encoding="latin-1")

    def _read_tsv(self, file) -> pd.DataFrame:
        """Read TSV files."""
        return pd.read_csv(file, sep="\t")

    def _read_excel(self, file) -> pd.DataFrame:
        """Read Excel files (.xlsx, .xls)."""
        return pd.read_excel(file, engine="openpyxl")

    def _read_json(self, file) -> pd.DataFrame:
        """Read JSON files (records or columnar format)."""
        try:
            return pd.read_json(file)
        except ValueError:
            file.seek(0)
            return pd.read_json(file, lines=True)

    def _read_parquet(self, file) -> pd.DataFrame:
        """Read Parquet files."""
        return pd.read_parquet(file)

    @staticmethod
    def get_file_info(uploaded_file) -> dict:
        """Extract metadata about the uploaded file."""
        return {
            "filename": uploaded_file.name,
            "size_bytes": uploaded_file.size,
            "size_readable": (
                f"{uploaded_file.size / 1024:.1f} KB"
                if uploaded_file.size < 1024 ** 2
                else f"{uploaded_file.size / (1024 ** 2):.1f} MB"
            ),
            "type": uploaded_file.name.split(".")[-1].lower(),
        }