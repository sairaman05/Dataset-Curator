"""Custom CSS styling for the app — injected via st.markdown."""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:wght@700;800&display=swap');

/* ── Global Reset ── */
.stApp {
    background: #0a0e1a;
    color: #c9d1d9;
    font-family: 'DM Sans', sans-serif;
}

/* Main content area */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1200px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1225 0%, #111832 50%, #0d1225 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.15);
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'DM Sans', sans-serif;
    color: #e2e8f0;
    letter-spacing: -0.02em;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span,
[data-testid="stSidebar"] .stMarkdown label {
    color: #94a3b8;
    font-size: 0.9rem;
}

/* Sidebar file uploader */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(99, 102, 241, 0.35);
    border-radius: 12px;
    padding: 0.5rem;
    background: rgba(99, 102, 241, 0.04);
    transition: border-color 0.3s ease, background 0.3s ease;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
    border-color: rgba(99, 102, 241, 0.6);
    background: rgba(99, 102, 241, 0.08);
}

/* ── Headings ── */
h1 {
    font-family: 'Playfair Display', serif !important;
    color: #f1f5f9 !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    font-size: 2.4rem !important;
}

h2, h3 {
    font-family: 'DM Sans', sans-serif !important;
    color: #e2e8f0 !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.06) 100%);
    border: 1px solid rgba(99, 102, 241, 0.18);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    transition: transform 0.2s ease, border-color 0.3s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.4);
}

[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500;
    font-size: 1.8rem !important;
}

[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: all 0.25s ease;
}

.stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border: none;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
}

.stButton > button[kind="primary"]:hover,
[data-testid="stSidebar"] .stButton > button:hover {
    box-shadow: 0 6px 25px rgba(99, 102, 241, 0.45);
    transform: translateY(-1px);
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 12px;
    overflow: hidden;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(99, 102, 241, 0.05);
    border-radius: 10px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    padding: 8px 20px;
}

.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.2) !important;
    color: #e2e8f0 !important;
}

/* ── Dividers ── */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.2), transparent);
    margin: 1.5rem 0;
}

/* ── Info/Success/Error boxes ── */
[data-testid="stAlert"] {
    border-radius: 12px;
    font-family: 'DM Sans', sans-serif;
}

/* ── Slider ── */
[data-testid="stSidebar"] .stSlider > div > div {
    color: #94a3b8;
}

/* ── Download buttons ── */
.stDownloadButton > button {
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.3);
    color: #4ade80;
    border-radius: 10px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
}

.stDownloadButton > button:hover {
    background: rgba(34, 197, 94, 0.2);
    border-color: rgba(34, 197, 94, 0.5);
}

/* ── Expander (cleaning log) ── */
[data-testid="stExpander"] {
    background: rgba(99, 102, 241, 0.04);
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 12px;
}

[data-testid="stExpander"] summary {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    color: #e2e8f0;
}

/* ── Section Cards (custom class via markdown) ── */
.section-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(15, 23, 42, 0.4));
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ── Welcome hero ── */
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.15));
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 0.8rem;
    color: #a5b4fc;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 1rem;
}

.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    color: #64748b;
    font-size: 1.1rem;
    line-height: 1.7;
    max-width: 700px;
}

/* ── Feature pills ── */
.feature-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 1.5rem;
}

.feature-pill {
    background: rgba(99, 102, 241, 0.06);
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 12px;
    padding: 14px 18px;
    transition: border-color 0.3s ease, transform 0.2s ease;
}

.feature-pill:hover {
    border-color: rgba(99, 102, 241, 0.3);
    transform: translateY(-1px);
}

.feature-pill .icon {
    font-size: 1.3rem;
    margin-bottom: 6px;
}

.feature-pill .label {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    color: #e2e8f0;
    font-size: 0.9rem;
}

.feature-pill .desc {
    font-family: 'DM Sans', sans-serif;
    color: #64748b;
    font-size: 0.78rem;
    margin-top: 2px;
}

/* ── Roadmap timeline ── */
.roadmap-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(99, 102, 241, 0.08);
}

.roadmap-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}

.roadmap-dot.active {
    background: #6366f1;
    box-shadow: 0 0 8px rgba(99, 102, 241, 0.5);
}

.roadmap-dot.upcoming {
    background: #334155;
    border: 1.5px solid #475569;
}

.roadmap-label {
    font-family: 'DM Sans', sans-serif;
    color: #94a3b8;
    font-size: 0.85rem;
}

.roadmap-label.active {
    color: #e2e8f0;
    font-weight: 600;
}

/* ── Cleaning log steps ── */
.log-step {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 14px;
    margin: 6px 0;
    background: rgba(99, 102, 241, 0.04);
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    color: #c9d1d9;
    line-height: 1.5;
}

.log-step-num {
    background: rgba(99, 102, 241, 0.2);
    color: #a5b4fc;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 6px;
    flex-shrink: 0;
}

/* ── Section headers with accent ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
}

.section-header .accent-bar {
    width: 4px;
    height: 24px;
    background: linear-gradient(180deg, #6366f1, #8b5cf6);
    border-radius: 2px;
}

.section-header .title {
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    color: #e2e8f0;
    font-size: 1.15rem;
    letter-spacing: -0.01em;
}

/* ── Scrollbar ── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: rgba(99, 102, 241, 0.25);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(99, 102, 241, 0.4);
}

/* ── Sprint 2: EDA Insight Cards ── */
[data-testid="stExpander"] details {
    border: none !important;
}

[data-testid="stExpander"] summary:hover {
    color: #a5b4fc !important;
}

/* Severity badge animations */
@keyframes pulse-critical {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

/* Top-level tab styling for Sprint 1 vs 2 */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(99, 102, 241, 0.04);
    border-radius: 12px;
    padding: 5px;
    border: 1px solid rgba(99, 102, 241, 0.08);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    color: #64748b;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    padding: 10px 24px;
    font-size: 0.9rem;
}

.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.15) !important;
    color: #e2e8f0 !important;
    border-bottom: none !important;
}

</style>
"""


def section_header(icon: str, title: str) -> str:
    """Generate an HTML section header with accent bar."""
    return f"""
    <div class="section-header">
        <div class="accent-bar"></div>
        <div class="title">{icon} {title}</div>
    </div>
    """