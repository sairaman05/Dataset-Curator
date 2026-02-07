"""
Curation Agent — orchestrates the full data pipeline.

This is the 'brain' that chains together:
    load → clean → profile → visualize
    
Future sprints will add:
    → feature engineering → model selection → training → evaluation
"""

import pandas as pd
import streamlit as st
from dataclasses import dataclass
from core.data_loader import DataLoader
from core.data_cleaner import DataCleaner, CleaningReport
from core.data_profiler import DataProfiler
from core.data_visualizer import DataVisualizer


@dataclass
class CurationResult:
    """Container for the full pipeline output."""
    raw_df: pd.DataFrame
    cleaned_df: pd.DataFrame
    report: CleaningReport
    profiler: DataProfiler
    visualizer: DataVisualizer
    file_info: dict


class CurationAgent:
    """
    Orchestrator agent for Sprint 1: Upload → Clean → Profile → Visualize.
    
    Usage:
        agent = CurationAgent()
        result = agent.run(uploaded_file)
    """

    def __init__(self, null_threshold: float = 0.7):
        self.loader = DataLoader()
        self.cleaner = DataCleaner(null_threshold=null_threshold)

    def run(self, uploaded_file) -> CurationResult:
        """
        Execute the full Sprint 1 pipeline.
        
        Args:
            uploaded_file: Streamlit UploadedFile
            
        Returns:
            CurationResult with all pipeline outputs
        """
        # Step 1: Load
        file_info = self.loader.get_file_info(uploaded_file)
        raw_df = self.loader.load(uploaded_file)

        # Step 2: Clean
        cleaned_df, report = self.cleaner.clean(raw_df)

        # Step 3: Profile
        profiler = DataProfiler(raw_df, cleaned_df, report)

        # Step 4: Visualize (on cleaned data)
        visualizer = DataVisualizer(cleaned_df)

        return CurationResult(
            raw_df=raw_df,
            cleaned_df=cleaned_df,
            report=report,
            profiler=profiler,
            visualizer=visualizer,
            file_info=file_info,
        )