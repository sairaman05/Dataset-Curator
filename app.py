"""
Dataset Curation Agent — Main Application
Sprint 1: Upload → Auto-Clean → Profile → Visualize
Sprint 2: Automated EDA → Insight Detection → Storytelling
Sprint 3: Feature Engineering → Interactive transforms, selection, export
Sprint 4: Auto ML Model Selection → Training → Evaluation
"""

import streamlit as st
from config.settings import APP_TITLE, APP_ICON, APP_LAYOUT
from agents.curation_agent import CurationAgent
from agents.eda_agent import EDAAgent
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
from ui.eda_view import (
    render_eda_executive_summary,
    render_eda_findings,
    render_eda_recommendations,
    render_eda_report_download,
)
from ui.fe_view import render_fe_tab
from ui.ml_view import render_ml_tab
from ui.styles import section_header
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
            <div class="hero-badge">Sprint 4 · Active</div>
            <h1 style="margin-bottom: 0.3rem;">DataCure !</h1>
            <p class="hero-subtitle">
                Upload your messy data — the agent handles cleaning, profiling, 
                insight detection, <strong style="color:#c084fc;">feature engineering</strong>,
                and <strong style="color:#a5b4fc;">ML model training</strong> automatically.
            </p>
            <div class="feature-grid">
                <div class="feature-pill">
                    <div class="icon">🧹</div>
                    <div class="label">Auto-Clean</div>
                    <div class="desc">Nulls, duplicates, dtypes — handled</div>
                </div>
                <div class="feature-pill">
                    <div class="icon">🔍</div>
                    <div class="label">Insight Detection</div>
                    <div class="desc">Outliers, correlations, imbalances</div>
                </div>
                <div class="feature-pill">
                    <div class="icon">⚙️</div>
                    <div class="label">Feature Engineering</div>
                    <div class="desc">Encoding, scaling, selection, interactions</div>
                </div>
                <div class="feature-pill">
                    <div class="icon">🤖</div>
                    <div class="label">Auto ML</div>
                    <div class="desc">Smart model selection & training</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Run Pipeline (Sprint 1 + 2) ──
if run_clicked or "result" in st.session_state:
    if run_clicked:
        with st.spinner("⚡ Cleaning & profiling your dataset..."):
            agent = CurationAgent(null_threshold=null_threshold)
            try:
                result = agent.run(uploaded_file)
                st.session_state["result"] = result
            except Exception as e:
                st.error(f"❌ Error during cleaning: {str(e)}")
                st.stop()

        with st.spinner("🔍 Detecting insights & generating story..."):
            try:
                eda_agent = EDAAgent()
                eda_result = eda_agent.run(
                    cleaned_df=result.cleaned_df,
                    raw_df=result.raw_df,
                    filename=result.file_info["filename"],
                )
                st.session_state["eda_result"] = eda_result
            except Exception as e:
                st.error(f"❌ Error during EDA: {str(e)}")

        # Clear FE + ML state on new upload
        for key in [
            "fe_agent", "fe_engineered_df",
            "ml_agent", "ml_analysis", "ml_recommendations", "ml_all_models",
            "ml_single_result", "ml_cmp_result_a", "ml_cmp_result_b", "ml_comparison",
        ]:
            st.session_state.pop(key, None)

        st.toast("Analysis complete!", icon="✅")

    if "result" in st.session_state:
        result = st.session_state["result"]

        # Title
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

        # ── Four Main Tabs ──
        tab_clean, tab_eda, tab_fe, tab_ml = st.tabs([
            "🧹 Clean & Profile",
            "🔍 EDA & Insights",
            "⚙️ Feature Engineering",
            "🤖 ML Training",
        ])

        with tab_clean:
            render_overview_metrics(result)
            st.markdown("<br>", unsafe_allow_html=True)
            render_cleaning_log(result)
            st.markdown("<br>", unsafe_allow_html=True)
            render_null_comparison(result)
            st.markdown("<br>", unsafe_allow_html=True)
            render_column_info(result)
            st.markdown("<br>", unsafe_allow_html=True)

            tab_num, tab_cat = st.tabs(["🔢 Numeric Stats", "🏷️ Categorical Stats"])
            with tab_num:
                render_numeric_stats(result)
            with tab_cat:
                render_categorical_stats(result)

            st.markdown("<br>", unsafe_allow_html=True)
            render_visualizations(result)
            st.markdown("<br>", unsafe_allow_html=True)
            render_data_preview(result)

            st.markdown("<br>", unsafe_allow_html=True)
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

        with tab_eda:
            if "eda_result" in st.session_state:
                eda_result = st.session_state["eda_result"]
                render_eda_executive_summary(eda_result)
                st.markdown("<br>", unsafe_allow_html=True)
                render_eda_findings(eda_result)
                st.markdown("<br>", unsafe_allow_html=True)
                render_eda_recommendations(eda_result)
                st.markdown("<br>", unsafe_allow_html=True)
                render_eda_report_download(eda_result)
            else:
                st.info("EDA results not available. Please re-run the analysis.")

        with tab_fe:
            render_fe_tab(result.cleaned_df)

        with tab_ml:
            # Use feature-engineered data if available, otherwise cleaned data
            ml_df = st.session_state.get("fe_engineered_df", result.cleaned_df)
            render_ml_tab(ml_df)