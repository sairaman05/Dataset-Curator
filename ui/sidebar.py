"""Sidebar UI: file upload, settings, and controls."""

import streamlit as st
from config.settings import SUPPORTED_FILE_TYPES, NULL_DROP_THRESHOLD


def render_sidebar() -> tuple:
    """
    Render the sidebar with upload widget and cleaning settings.
    
    Returns:
        Tuple of (uploaded_file, null_threshold, run_clicked)
    """
    with st.sidebar:
        # Logo / branding area
        st.markdown(
            """
            <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
                <div style="
                    font-family: 'Playfair Display', serif;
                    font-size: 1.5rem;
                    font-weight: 800;
                    color: #f1f5f9;
                    letter-spacing: -0.03em;
                    line-height: 1.2;
                ">✦ Curation Agent</div>
                <div style="
                    font-family: 'DM Sans', sans-serif;
                    font-size: 0.72rem;
                    color: #64748b;
                    text-transform: uppercase;
                    letter-spacing: 0.1em;
                    margin-top: 4px;
                ">Automated Data Pipeline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<hr style="margin: 0.5rem 0 1rem 0; opacity: 0.3;">',
            unsafe_allow_html=True,
        )

        # Upload section
        st.markdown(
            """
            <div style="
                font-family: 'DM Sans', sans-serif;
                font-size: 0.75rem;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                font-weight: 600;
                margin-bottom: 8px;
            ">📂 Dataset Upload</div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=SUPPORTED_FILE_TYPES,
            help=f"Supported: {', '.join(SUPPORTED_FILE_TYPES)}",
            label_visibility="collapsed",
        )

        if uploaded_file:
            size = uploaded_file.size
            size_str = (
                f"{size / 1024:.1f} KB"
                if size < 1024 ** 2
                else f"{size / (1024 ** 2):.1f} MB"
            )
            ext = uploaded_file.name.split(".")[-1].upper()
            st.markdown(
                f"""
                <div style="
                    background: rgba(99, 102, 241, 0.08);
                    border: 1px solid rgba(99, 102, 241, 0.2);
                    border-radius: 10px;
                    padding: 10px 14px;
                    margin: 8px 0;
                ">
                    <div style="
                        font-family: 'DM Sans', sans-serif;
                        font-weight: 600;
                        color: #e2e8f0;
                        font-size: 0.85rem;
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                    ">📄 {uploaded_file.name}</div>
                    <div style="
                        font-family: 'JetBrains Mono', monospace;
                        color: #64748b;
                        font-size: 0.72rem;
                        margin-top: 4px;
                    ">{ext} · {size_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            '<hr style="margin: 1rem 0; opacity: 0.15;">',
            unsafe_allow_html=True,
        )

        # Settings section
        st.markdown(
            """
            <div style="
                font-family: 'DM Sans', sans-serif;
                font-size: 0.75rem;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                font-weight: 600;
                margin-bottom: 8px;
            ">⚙️ Cleaning Config</div>
            """,
            unsafe_allow_html=True,
        )

        null_threshold = st.slider(
            "Null column drop threshold",
            min_value=0.1,
            max_value=1.0,
            value=NULL_DROP_THRESHOLD,
            step=0.05,
            help="Columns with null % above this are dropped",
        )

        st.markdown(
            '<hr style="margin: 1rem 0; opacity: 0.15;">',
            unsafe_allow_html=True,
        )

        # Run button
        run_clicked = st.button(
            "⚡ Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
        )

        # Sprint roadmap
        st.markdown(
            '<hr style="margin: 1.2rem 0; opacity: 0.15;">',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="
                font-family: 'DM Sans', sans-serif;
                font-size: 0.75rem;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                font-weight: 600;
                margin-bottom: 10px;
            ">🗺️ Roadmap</div>
            """
            + _roadmap_html(),
            unsafe_allow_html=True,
        )

    return uploaded_file, null_threshold, run_clicked


def _roadmap_html() -> str:
    """Generate the sprint roadmap HTML."""
    sprints = [
        ("Clean & Profile", True),
        ("EDA Storytelling", True),
        ("ML Training", True),
        ("Feature Engineering", False),
        ("Deep Learning", False),
        ("Results Export", False),
    ]
    items = ""
    for label, active in sprints:
        dot_cls = "active" if active else "upcoming"
        lbl_cls = "active" if active else ""
        items += f"""
        <div class="roadmap-item">
            <div class="roadmap-dot {dot_cls}"></div>
            <div class="roadmap-label {lbl_cls}">{label}</div>
        </div>
        """
    return items