"""EDA storytelling display components."""

import streamlit as st
from agents.eda_agent import EDAResult
from core.insight_engine import Severity
from ui.styles import section_header


SEVERITY_COLORS = {
    Severity.CRITICAL: ("#fca5a5", "#7f1d1d", "#ef4444", "rgba(239,68,68,0.08)"),
    Severity.WARNING: ("#fde68a", "#78350f", "#f59e0b", "rgba(245,158,11,0.08)"),
    Severity.INFO: ("#93c5fd", "#1e3a5f", "#6366f1", "rgba(99,102,241,0.08)"),
}


def render_eda_executive_summary(eda_result: EDAResult):
    """Render the executive summary banner."""
    st.markdown(section_header("📝", "Executive Summary"), unsafe_allow_html=True)

    # Severity counts
    critical = sum(1 for i in eda_result.insights if i.severity == Severity.CRITICAL)
    warnings = sum(1 for i in eda_result.insights if i.severity == Severity.WARNING)
    info = sum(1 for i in eda_result.insights if i.severity == Severity.INFO)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Insights", len(eda_result.insights))
    with col2:
        st.metric("🔴 Critical", critical)
    with col3:
        st.metric("🟡 Warnings", warnings)
    with col4:
        st.metric("🔵 Info", info)

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(99,102,241,0.06), rgba(139,92,246,0.04));
            border: 1px solid rgba(99,102,241,0.15);
            border-radius: 14px;
            padding: 1.2rem 1.5rem;
            margin-top: 0.8rem;
            font-family: 'DM Sans', sans-serif;
            font-size: 0.95rem;
            color: #c9d1d9;
            line-height: 1.7;
        ">{eda_result.executive_summary}</div>
        """,
        unsafe_allow_html=True,
    )


def render_eda_findings(eda_result: EDAResult):
    """Render organized insight findings as styled cards."""
    st.markdown(section_header("🔍", "Detailed Findings"), unsafe_allow_html=True)

    for section in eda_result.findings:
        with st.expander(f"📂 {section['section']}", expanded=True):
            for insight in section["insights"]:
                _render_insight_card(insight)


def _render_insight_card(insight: dict):
    """Render a single insight as a styled card."""
    severity = insight["severity"]
    text_color, _, border_color, bg_color = SEVERITY_COLORS.get(
        severity, SEVERITY_COLORS[Severity.INFO]
    )

    cols_html = ""
    if insight["affected_columns"]:
        pills = "".join(
            f'<span style="'
            f"background: rgba(99,102,241,0.1); "
            f"border: 1px solid rgba(99,102,241,0.2); "
            f"border-radius: 6px; "
            f"padding: 2px 8px; "
            f"font-family: 'JetBrains Mono', monospace; "
            f"font-size: 0.72rem; "
            f"color: #a5b4fc; "
            f"margin-right: 4px; "
            f'">{col}</span>'
            for col in insight["affected_columns"][:8]
        )
        cols_html = f'<div style="margin-top: 8px;">{pills}</div>'

    rec_html = ""
    if insight["recommendation"]:
        rec_html = f"""
        <div style="
            margin-top: 10px;
            padding: 8px 12px;
            background: rgba(34,197,94,0.06);
            border-left: 3px solid #22c55e;
            border-radius: 0 8px 8px 0;
            font-size: 0.82rem;
            color: #86efac;
        ">💡 {insight['recommendation']}</div>
        """

    st.markdown(
        f"""
        <div style="
            background: {bg_color};
            border: 1px solid {border_color}22;
            border-left: 4px solid {border_color};
            border-radius: 0 12px 12px 0;
            padding: 14px 18px;
            margin: 8px 0;
        ">
            <div style="
                font-family: 'DM Sans', sans-serif;
                font-weight: 700;
                color: #e2e8f0;
                font-size: 0.95rem;
                margin-bottom: 6px;
            ">{insight['severity_emoji']} {insight['title']}</div>
            <div style="
                font-family: 'DM Sans', sans-serif;
                color: #94a3b8;
                font-size: 0.85rem;
                line-height: 1.6;
            ">{insight['description']}</div>
            {cols_html}
            {rec_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_eda_recommendations(eda_result: EDAResult):
    """Render prioritized recommendations."""
    recs = eda_result.recommendations
    if not recs:
        return

    st.markdown(section_header("🎯", "Prioritized Recommendations"), unsafe_allow_html=True)

    for i, rec in enumerate(recs, 1):
        severity = rec["severity"]
        _, _, border_color, bg_color = SEVERITY_COLORS.get(
            severity, SEVERITY_COLORS[Severity.INFO]
        )

        st.markdown(
            f"""
            <div style="
                display: flex;
                align-items: flex-start;
                gap: 12px;
                padding: 12px 16px;
                margin: 6px 0;
                background: {bg_color};
                border: 1px solid {border_color}18;
                border-radius: 10px;
            ">
                <div style="
                    background: {border_color}25;
                    color: {border_color};
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 0.75rem;
                    font-weight: 700;
                    padding: 3px 10px;
                    border-radius: 6px;
                    flex-shrink: 0;
                ">{i:02d}</div>
                <div>
                    <div style="
                        font-family: 'DM Sans', sans-serif;
                        font-weight: 600;
                        color: #e2e8f0;
                        font-size: 0.85rem;
                    ">{rec['context']}</div>
                    <div style="
                        font-family: 'DM Sans', sans-serif;
                        color: #94a3b8;
                        font-size: 0.82rem;
                        margin-top: 3px;
                        line-height: 1.5;
                    ">{rec['recommendation']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_eda_report_download(eda_result: EDAResult):
    """Download button for the markdown report."""
    st.markdown(section_header("📄", "Export EDA Report"), unsafe_allow_html=True)

    st.download_button(
        label="⬇ Download Full EDA Report (.md)",
        data=eda_result.markdown_report.encode("utf-8"),
        file_name="eda_report.md",
        mime="text/markdown",
        use_container_width=True,
    )