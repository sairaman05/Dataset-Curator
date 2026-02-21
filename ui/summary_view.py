"""Summary and profiling display components."""

import streamlit as st
import pandas as pd
from agents.curation_agent import CurationResult
from utils.helpers import format_number, get_memory_usage
from ui.styles import section_header


def render_overview_metrics(result: CurationResult):
    """Render top-level KPI metrics."""
    overview = result.profiler.get_overview()

    st.markdown(section_header("📊", "Dataset Overview"), unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Rows",
            format_number(overview["rows_after"]),
            delta=f"-{overview['rows_removed']}" if overview["rows_removed"] else "0",
            delta_color="normal",
        )
    with col2:
        st.metric(
            "Columns",
            format_number(overview["cols_after"]),
            delta=f"-{overview['cols_removed']}" if overview["cols_removed"] else "0",
            delta_color="normal",
        )
    with col3:
        st.metric(
            "Nulls Fixed",
            format_number(overview["nulls_resolved"]),
            delta=f"{overview['total_nulls_before']} → {overview['total_nulls_after']}",
            delta_color="off",
        )
    with col4:
        st.metric(
            "Dupes Removed",
            format_number(overview["duplicates_removed"]),
        )

    # Sub-info
    st.markdown(
        f"""
        <div style="
            display: flex;
            gap: 24px;
            margin-top: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: #475569;
        ">
            <span>💾 {get_memory_usage(result.cleaned_df)}</span>
            <span>📁 {result.file_info['filename']}</span>
            <span>📐 {result.file_info['size_readable']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_cleaning_log(result: CurationResult):
    """Display the step-by-step cleaning actions log."""
    st.markdown(section_header("🧹", "Cleaning Actions"), unsafe_allow_html=True)

    if result.report.actions_log:
        log_html = ""
        for i, action in enumerate(result.report.actions_log, 1):
            log_html += f"""
            <div class="log-step">
                <span class="log-step-num">{i:02d}</span>
                <span>{action}</span>
            </div>
            """
        with st.expander(f"View all {len(result.report.actions_log)} cleaning steps", expanded=True):
            st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.success("Your data was already clean — no actions needed!")


def render_column_info(result: CurationResult):
    """Display per-column information table."""
    st.markdown(section_header("📋", "Column Information"), unsafe_allow_html=True)

    col_info = result.profiler.get_column_info()
    st.dataframe(col_info, use_container_width=True, hide_index=True)


def render_numeric_stats(result: CurationResult):
    """Display descriptive stats for numeric columns."""
    stats = result.profiler.get_numeric_stats()
    if not stats.empty:
        st.markdown(section_header("🔢", "Numeric Statistics"), unsafe_allow_html=True)
        st.dataframe(stats, use_container_width=True)
    else:
        st.info("No numeric columns found in the dataset.")


def render_categorical_stats(result: CurationResult):
    """Display stats for categorical columns."""
    stats = result.profiler.get_categorical_stats()
    if not stats.empty:
        st.markdown(section_header("🏷️", "Categorical Statistics"), unsafe_allow_html=True)
        st.dataframe(stats, use_container_width=True, hide_index=True)
    else:
        st.info("No categorical columns found in the dataset.")


def render_data_preview(result: CurationResult):
    """Tabbed preview of raw vs cleaned data."""
    st.markdown(section_header("👀", "Data Preview"), unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["✨ Cleaned Data", "📄 Raw Data"])
    with tab1:
        st.dataframe(result.cleaned_df.head(100), use_container_width=True)
    with tab2:
        st.dataframe(result.raw_df.head(100), use_container_width=True)


def render_null_comparison(result: CurationResult):
    """Null before vs after table."""
    null_df = result.profiler.get_null_comparison()
    has_nulls = null_df[null_df["Nulls Before"] > 0]
    if not has_nulls.empty:
        st.markdown(section_header("🕳️", "Null Value Comparison"), unsafe_allow_html=True)
        st.dataframe(has_nulls, use_container_width=True, hide_index=True)