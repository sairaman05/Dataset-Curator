"""
Dataset Curation Agent — Main Application
Sprint 1: Upload → Auto-Clean → Profile → Visualize
"""

import streamlit as st
from config.settings import APP_TITLE, APP_ICON, APP_LAYOUT
from agents.curation_agent import CurationAgent
from ui.styles import CUSTOM_CSS
from ui.sidebar import render_sidebar
from ui.summary_view import (
    render_overview_metrics,
    render_cleaning_log,
    render_column_info,
    render_numeric_stats,
    render_categorical_stats,
    render_data_preview,
    render_null_comparison,
)
from ui.visualization_view import render_visualizations
from utils.helpers import dataframe_to_excel_bytes


# Page config
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
)

# Inject custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Sidebar
uploaded_file, null_threshold, run_clicked = render_sidebar()

# ── Welcome Screen ──
if not uploaded_file:
    st.markdown(
        """
        <div style="padding: 2rem 0;">
            <div class="hero-badge">Sprint 1 · Active</div>
            <h1 style="margin-bottom: 0.3rem;">Dataset Curation Agent</h1>
            <p class="hero-subtitle">
                Upload your messy data — the agent handles cleaning, profiling, and visualization 
                automatically. No code required. Just insights.
            </p>
            <div class="feature-grid">
                <div class="feature-pill">
                    <div class="icon">🧹</div>
                    <div class="label">Auto-Clean</div>
                    <div class="desc">Nulls, duplicates, dtypes — handled</div>
                </div>
                <div class="feature-pill">
                    <div class="icon">📊</div>
                    <div class="label">Smart Profiling</div>
                    <div class="desc">Stats & column analysis in seconds</div>
                </div>
                <div class="feature-pill">
                    <div class="icon">📈</div>
                    <div class="label">Auto Visualize</div>
                    <div class="desc">Distributions, correlations, comparisons</div>
                </div>
                <div class="feature-pill">
                    <div class="icon">💾</div>
                    <div class="label">Export Clean Data</div>
                    <div class="desc">Download as CSV or Excel</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Run Pipeline ──
if run_clicked or "result" in st.session_state:
    if run_clicked:
        with st.spinner("⚡ Agent is analyzing your dataset..."):
            agent = CurationAgent(null_threshold=null_threshold)
            try:
                result = agent.run(uploaded_file)
                st.session_state["result"] = result
                st.toast("Analysis complete!", icon="✅")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.stop()

    if "result" in st.session_state:
        result = st.session_state["result"]

        # Title for results
        st.markdown(
            f"""
            <div style="margin-bottom: 1rem;">
                <h1 style="font-size: 1.8rem !important; margin-bottom: 0;">Analysis Results</h1>
                <span style="
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 0.78rem;
                    color: #475569;
                ">{result.file_info['filename']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_overview_metrics(result)
        st.markdown("<br>", unsafe_allow_html=True)
        render_cleaning_log(result)
        st.markdown("<br>", unsafe_allow_html=True)
        render_null_comparison(result)
        st.markdown("<br>", unsafe_allow_html=True)
        render_column_info(result)

        st.markdown("<br>", unsafe_allow_html=True)

        # Stats tabs
        tab_num, tab_cat = st.tabs(["🔢 Numeric Stats", "🏷️ Categorical Stats"])
        with tab_num:
            render_numeric_stats(result)
        with tab_cat:
            render_categorical_stats(result)

        st.markdown("<br>", unsafe_allow_html=True)
        render_visualizations(result)
        st.markdown("<br>", unsafe_allow_html=True)
        render_data_preview(result)

        # Download section
        st.markdown("<br>", unsafe_allow_html=True)

        from ui.styles import section_header
        st.markdown(section_header("💾", "Export Cleaned Data"), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            csv_data = result.cleaned_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Download CSV",
                data=csv_data,
                file_name="cleaned_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col2:
            excel_data = dataframe_to_excel_bytes(result.cleaned_df)
            st.download_button(
                label="⬇ Download Excel",
                data=excel_data,
                file_name="cleaned_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )