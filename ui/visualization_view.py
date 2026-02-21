"""Visualization display components."""

import streamlit as st
from agents.curation_agent import CurationResult
from ui.styles import section_header


# Dark plotly template override
DARK_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10, 14, 26, 0.6)",
        font=dict(family="DM Sans, sans-serif", color="#94a3b8"),
        title=dict(font=dict(family="DM Sans, sans-serif", color="#e2e8f0", size=16)),
        xaxis=dict(gridcolor="rgba(99,102,241,0.08)", zerolinecolor="rgba(99,102,241,0.15)"),
        yaxis=dict(gridcolor="rgba(99,102,241,0.08)", zerolinecolor="rgba(99,102,241,0.15)"),
        colorway=["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#06b6d4", "#ec4899",
                   "#8b5cf6", "#14b8a6", "#f97316", "#64748b"],
    )
)


def _apply_dark_theme(fig):
    """Apply consistent dark theme to a plotly figure."""
    if fig is None:
        return None
    fig.update_layout(**DARK_TEMPLATE["layout"])
    return fig


def render_visualizations(result: CurationResult):
    """Render all auto-generated visualizations."""
    st.markdown(section_header("📈", "Visualizations"), unsafe_allow_html=True)

    profiler = result.profiler
    viz = result.visualizer

    # Null comparison chart
    null_df = profiler.get_null_comparison()
    null_chart = _apply_dark_theme(viz.null_comparison_chart(null_df))
    if null_chart:
        st.plotly_chart(null_chart, use_container_width=True)

    # Two-column: dtype pie + correlation heatmap
    col1, col2 = st.columns(2)
    with col1:
        dtype_chart = _apply_dark_theme(viz.dtype_distribution_chart())
        if dtype_chart:
            st.plotly_chart(dtype_chart, use_container_width=True)

    with col2:
        corr_chart = _apply_dark_theme(viz.correlation_heatmap())
        if corr_chart:
            st.plotly_chart(corr_chart, use_container_width=True)

    # Numeric distributions
    num_chart = _apply_dark_theme(viz.numeric_distributions())
    if num_chart:
        st.plotly_chart(num_chart, use_container_width=True)

    # Categorical distributions
    cat_chart = _apply_dark_theme(viz.categorical_distributions())
    if cat_chart:
        st.plotly_chart(cat_chart, use_container_width=True)