# ui/fe_view.py
"""
Feature Engineering Tab — Complete UI matching 5-category structure.

① Feature Creation      — Polynomial (synthetic), Arithmetic (combination), Aggregation (row-wise)
② Feature Transformation — Encoding (Label/OHE/Target/Freq) + Math (Log/Sqrt/BoxCox/Binning)
③ Feature Extraction    — PCA with full viz, DateTime extraction
④ Feature Selection     — 🚀 AUTO BUTTON + Filter/Wrapper/Embedded with intermediate viz
⑤ Feature Scaling       — Standard, MinMax, Robust, MaxAbs
⑥ Drop Columns
⑦ Operation Log + Undo/Reset
⑧ Preview + Download + Send to ML
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict

from agents.fe_agent import FEAgent


# ═══════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════

FE_CSS = """
<style>
.fe-hdr {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem; font-weight: 700; color: #e2e8f0;
    padding: 0.4rem 0; border-bottom: 2px solid #8b5cf6;
    margin: 1.6rem 0 0.7rem 0;
}
.fe-m {
    background: linear-gradient(135deg, #0f1320, #1a1f35);
    border: 1px solid #2d3555; border-radius: 10px;
    padding: 0.65rem; text-align: center;
}
.fe-m .lb { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
.fe-m .vl { font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 700; color: #c7d2fe; }
.fe-m .dp { color: #6ee7b7; font-size: 0.75rem; }
.fe-m .dn { color: #fca5a5; font-size: 0.75rem; }
.fe-m .dz { color: #94a3b8; font-size: 0.75rem; }
.fe-log {
    background: #0f1320; border-left: 3px solid #8b5cf6;
    padding: 0.5rem 0.8rem; margin-bottom: 0.4rem;
    border-radius: 0 6px 6px 0; font-size: 0.85rem;
}
.fe-badge {
    display: inline-block; font-size: 0.63rem; font-weight: 600;
    padding: 1px 6px; border-radius: 4px; margin-left: 6px; text-transform: uppercase;
}
.fe-cat-creation { background: #1e3a5f; color: #7dd3fc; }
.fe-cat-transformation { background: #3b1f5e; color: #c4b5fd; }
.fe-cat-extraction { background: #1a3c34; color: #6ee7b7; }
.fe-cat-selection { background: #5c3d1e; color: #fcd34d; }
.fe-cat-scaling { background: #3b1f3b; color: #f9a8d4; }
.fe-cat-utility { background: #1e293b; color: #94a3b8; }
.fe-ok {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981; border-radius: 8px;
    padding: 0.6rem 1rem; margin: 0.5rem 0; color: #a7f3d0; font-size: 0.9rem;
}
.fe-auto {
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    border: 2px solid #6366f1; border-radius: 12px;
    padding: 1rem 1.2rem; margin: 0.8rem 0;
}
.fe-auto h3 { color: #a5b4fc; margin: 0 0 0.3rem 0; font-size: 1.1rem; }
.fe-auto p { color: #c7d2fe; margin: 0; font-size: 0.85rem; }
</style>
"""

# ─── helpers ───
def _h(t):
    st.markdown(f'<div class="fe-hdr">{t}</div>', unsafe_allow_html=True)

def _m(col, label, val, delta=None):
    dh = ""
    if delta is not None and delta != 0:
        c = "dp" if delta > 0 else "dn"
        s = "+" if delta > 0 else ""
        dh = f'<div class="{c}">{s}{delta}</div>'
    elif delta == 0:
        dh = '<div class="dz">—</div>'
    col.markdown(f'<div class="fe-m"><div class="lb">{label}</div><div class="vl">{val}</div>{dh}</div>',
                 unsafe_allow_html=True)

def _ok(t):
    st.markdown(f'<div class="fe-ok">✅ {t}</div>', unsafe_allow_html=True)

def _badge(cat):
    return f'<span class="fe-badge fe-cat-{cat}">{cat}</span>'

def _agent(df):
    if "fe_agent" not in st.session_state:
        st.session_state.fe_agent = FEAgent(df)
    return st.session_state.fe_agent


# ═══════════════════════════════════════════════════════
# PLOTLY CHARTS FOR INTERMEDIATES
# ═══════════════════════════════════════════════════════

def _bar(scores: dict, title: str, xlabel: str = "Score", color: str = "#8b5cf6"):
    s = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fig = go.Figure(go.Bar(
        x=[v for _, v in s], y=[n for n, _ in s], orientation="h",
        marker_color=color, text=[f"{v:.4f}" for _, v in s], textposition="outside",
    ))
    fig.update_layout(
        title=title, height=max(300, len(s) * 28),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=xlabel, yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=40, b=30), font=dict(size=11),
    )
    return fig


def _heatmap(corr_dict: dict, title: str = "Correlation Matrix"):
    cdf = pd.DataFrame(corr_dict)
    fig = go.Figure(go.Heatmap(
        z=cdf.values, x=cdf.columns.tolist(), y=cdf.index.tolist(),
        colorscale="RdBu_r", zmid=0, text=np.round(cdf.values, 2), texttemplate="%{text}",
        textfont={"size": 9},
    ))
    n = len(cdf)
    fig.update_layout(
        title=title, height=max(400, n * 30 + 100),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10), font=dict(size=10),
    )
    return fig


def _pca_variance(info: dict):
    evr = info["explained_variance_ratio"]
    cum = info["cumulative_variance"]
    pcs = [f"PC{i+1}" for i in range(len(evr))]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=pcs, y=[v*100 for v in evr], name="Individual %",
        marker_color="#8b5cf6", text=[f"{v:.1%}" for v in evr], textposition="outside",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=pcs, y=[v*100 for v in cum], name="Cumulative %",
        mode="lines+markers", line=dict(color="#f59e0b", width=2.5), marker=dict(size=8),
    ), secondary_y=True)
    fig.update_layout(
        title="PCA — Explained Variance", height=380,
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=30), font=dict(size=11),
        legend=dict(orientation="h", y=1.12),
    )
    fig.update_yaxes(title_text="Individual %", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 105])
    return fig


def _pca_loadings(info: dict, pc: str):
    if "top_features_per_pc" in info and pc in info["top_features_per_pc"]:
        return _bar(info["top_features_per_pc"][pc], f"Top Contributors → {pc}", "|Loading|", "#6366f1")
    return None


def _auto_rankings(info: dict):
    rankings = info.get("method_rankings", {})
    avg = info.get("average_ranks", {})
    if not rankings or not avg:
        return None

    features = list(avg.keys())
    methods = list(rankings.keys())
    palette = ["#8b5cf6", "#f59e0b", "#10b981", "#ef4444", "#3b82f6"]

    fig = go.Figure()
    for i, method in enumerate(methods):
        mr = rankings[method]
        fig.add_trace(go.Bar(
            name=method, x=features,
            y=[mr.get(f, len(features)) for f in features],
            marker_color=palette[i % len(palette)], opacity=0.8,
        ))
    fig.add_trace(go.Scatter(
        name="Avg Rank", x=features,
        y=[avg.get(f, len(features)) for f in features],
        mode="lines+markers", line=dict(color="white", width=3),
        marker=dict(size=10, symbol="diamond"),
    ))
    fig.update_layout(
        title="Feature Rankings Across All Methods (lower = better)",
        barmode="group", height=460,
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=30), font=dict(size=10),
        yaxis_title="Rank (1=best)", legend=dict(orientation="h", y=1.15),
        xaxis_tickangle=-45,
    )
    return fig


# ═══════════════════════════════════════════════════════
# MAIN RENDER
# ═══════════════════════════════════════════════════════

def render_fe_tab(cleaned_df: Optional[pd.DataFrame]):
    st.markdown(FE_CSS, unsafe_allow_html=True)
    if cleaned_df is None:
        st.info("⬆️ Upload and clean a dataset first.")
        return

    agent = _agent(cleaned_df)
    num_cols = agent.numeric_columns
    cat_cols = agent.categorical_columns
    dt_cols = agent.datetime_columns

    # ══════════════════════════════════════════════
    # OVERVIEW
    # ══════════════════════════════════════════════
    _h("📊 Dataset Overview")
    s = agent.get_comparison_stats()
    c1, c2, c3, c4, c5 = st.columns(5)
    _m(c1, "Columns", s["columns"]["after"], s["columns"]["after"] - s["columns"]["before"])
    _m(c2, "Rows", f'{s["rows"]["after"]:,}')
    _m(c3, "Numeric", s["numeric"]["after"], s["numeric"]["after"] - s["numeric"]["before"])
    _m(c4, "Categorical", s["categorical"]["after"], s["categorical"]["after"] - s["categorical"]["before"])
    _m(c5, "Memory", f'{s["memory_mb"]["after"]}MB')
    if agent.logs:
        st.caption(f"🔧 {len(agent.logs)} transforms applied")

    # ══════════════════════════════════════════════
    # ① FEATURE CREATION
    # ══════════════════════════════════════════════
    _h("① Feature Creation")
    st.caption("Generate new features: synthetic combinations, arithmetic operations, row-wise aggregations.")

    if len(num_cols) < 2:
        st.info("Need at least 2 numeric columns for feature creation.")
    else:
        cr1, cr2, cr3 = st.tabs(["🔢 Polynomial (Synthetic)", "➕ Arithmetic (Combination)", "📊 Aggregation"])

        with cr1:
            st.markdown("Creates new features by combining existing features polynomially (e.g., `age²`, `age×income`).")
            poly_cols = st.multiselect("Columns", num_cols,
                                        default=num_cols[:3] if len(num_cols) >= 3 else num_cols,
                                        key="fe_poly_c")
            p1, p2 = st.columns(2)
            poly_d = p1.slider("Degree", 2, 3, 2, key="fe_poly_d")
            poly_i = p2.checkbox("Interaction only (no x², only x×y)", key="fe_poly_i")
            if st.button("▶ Generate Polynomial Features", key="btn_poly", disabled=not poly_cols):
                log, _ = agent.apply_polynomial(poly_cols, degree=poly_d, interaction_only=poly_i)
                _ok(log.detail)
                st.rerun()

        with cr2:
            st.markdown("Combines two columns with arithmetic: `A+B`, `A−B`, `A×B`, `A÷B`.")
            a1, a2 = st.columns(2)
            ca = a1.selectbox("Column A", num_cols, key="fe_ar_a")
            rem = [c for c in num_cols if c != ca]
            cb = a2.selectbox("Column B", rem if rem else num_cols, key="fe_ar_b")
            ar_ops = st.multiselect("Operations", ["add", "subtract", "multiply", "divide"],
                                     default=["add", "multiply"], key="fe_ar_ops")
            if st.button("▶ Create Arithmetic Features", key="btn_arith", disabled=not ar_ops):
                log, _ = agent.apply_arithmetic(ca, cb, operations=ar_ops)
                _ok(log.detail)
                st.rerun()

        with cr3:
            st.markdown("Row-wise summary statistics across selected columns (mean, sum, std, min, max, range).")
            agg_cols = st.multiselect("Columns", num_cols, key="fe_agg_c")
            agg_fn = st.multiselect("Functions", ["mean", "sum", "std", "min", "max", "range"],
                                     default=["mean", "sum", "std"], key="fe_agg_fn")
            if st.button("▶ Create Aggregation Features", key="btn_agg", disabled=not agg_cols or not agg_fn):
                log, _ = agent.apply_aggregation(agg_cols, agg_funcs=agg_fn)
                _ok(log.detail)
                st.rerun()

    # ══════════════════════════════════════════════
    # ② FEATURE TRANSFORMATION
    # ══════════════════════════════════════════════
    _h("② Feature Transformation")
    st.caption("Adjust features for better model learning: encode categoricals, apply math transforms.")

    t1, t2 = st.tabs(["🏷️ Encoding (Categorical → Numeric)", "📐 Mathematical Transforms"])

    with t1:
        if not cat_cols:
            st.info("No categorical columns detected.")
        else:
            enc = st.selectbox("Method", [
                "One-Hot Encoding", "Label Encoding", "Frequency Encoding", "Target Encoding"
            ], key="fe_enc")
            ec = st.multiselect("Columns", cat_cols, key="fe_ec")
            ek = {}
            if enc == "One-Hot Encoding":
                e1, e2 = st.columns(2)
                ek["drop_first"] = e1.checkbox("Drop first dummy", value=True, key="fe_ohe_d")
                ek["max_categories"] = e2.slider("Max categories", 5, 50, 20, key="fe_ohe_m")
            if enc == "Target Encoding":
                ek["target_col"] = st.selectbox("Target column", agent.columns, key="fe_te_t")

            if st.button("▶ Apply Encoding", key="btn_enc", disabled=not ec):
                if enc == "One-Hot Encoding":
                    log, _ = agent.apply_onehot_encoding(ec, **ek)
                elif enc == "Label Encoding":
                    log, _ = agent.apply_label_encoding(ec)
                elif enc == "Frequency Encoding":
                    log, _ = agent.apply_frequency_encoding(ec)
                elif enc == "Target Encoding":
                    log, _ = agent.apply_target_encoding(ec, **ek)
                _ok(log.detail)
                if hasattr(log, 'intermediates') and log.intermediates and "mappings" in log.intermediates:
                    with st.expander("📋 Label Mappings"):
                        st.json(log.intermediates["mappings"])
                st.rerun()

    with t2:
        if not num_cols:
            st.info("No numeric columns.")
        else:
            mt = st.selectbox("Transform", [
                "Log (log1p) — reduces right skew",
                "Square Root — milder than log",
                "Box-Cox (Yeo-Johnson) — optimal power transform",
                "Binning / Discretize — numeric → bins",
            ], key="fe_mt")
            mc = st.multiselect("Columns", num_cols, key="fe_mc")
            mk = {}
            if "Binning" in mt:
                b1, b2 = st.columns(2)
                mk["n_bins"] = b1.slider("Bins", 2, 10, 5, key="fe_bn")
                mk["strategy"] = b2.selectbox("Strategy", ["quantile", "uniform"], key="fe_bs")

            if st.button("▶ Apply Transform", key="btn_mt", disabled=not mc):
                if "Log" in mt:
                    log, _ = agent.apply_log_transform(mc)
                elif "Square" in mt:
                    log, _ = agent.apply_sqrt_transform(mc)
                elif "Box-Cox" in mt:
                    log, _ = agent.apply_boxcox_transform(mc)
                elif "Binning" in mt:
                    log, _ = agent.apply_binning(mc, **mk)
                _ok(log.detail)
                if hasattr(log, 'intermediates') and log.intermediates and "lambdas" in (log.intermediates or {}):
                    with st.expander("📋 Yeo-Johnson Lambdas"):
                        st.json(log.intermediates["lambdas"])
                st.rerun()

    # ══════════════════════════════════════════════
    # ③ FEATURE EXTRACTION
    # ══════════════════════════════════════════════
    _h("③ Feature Extraction")
    st.caption("Reduce dimensionality while preserving information. Extract meaningful components from raw features.")

    ex1, ex2 = st.tabs(["📉 PCA / Dimensionality Reduction", "📅 DateTime Extraction"])

    with ex1:
        if len(num_cols) < 2:
            st.info("Need ≥ 2 numeric columns for PCA.")
        else:
            st.markdown("""
            **PCA** (Principal Component Analysis) finds new axes that capture the most variance in your data.
            It reduces many features to fewer components while keeping as much information as possible.
            """)
            pca_cols = st.multiselect("Columns for PCA", num_cols, default=num_cols, key="fe_pca_c")
            pca_n = st.slider("Number of components", 2,
                               min(20, len(pca_cols)) if pca_cols else 2,
                               min(5, len(pca_cols)) if pca_cols else 2, key="fe_pca_n")

            if st.button("▶ Run PCA", key="btn_pca", type="primary", disabled=len(pca_cols) < 2):
                with st.spinner("Running PCA..."):
                    log, info = agent.apply_pca(pca_cols, n_components=pca_n)
                _ok(log.detail)

                if info:
                    # Chart 1: Explained variance bar + cumulative line
                    st.plotly_chart(_pca_variance(info), use_container_width=True)

                    # Chart 2: Top feature contributions per PC
                    st.markdown("**Feature Contributions (Loadings) — which original features drive each component:**")
                    pc_tabs = st.tabs([f"PC{i+1}" for i in range(info["n_components"])])
                    for i, pc_tab in enumerate(pc_tabs):
                        with pc_tab:
                            fig = _pca_loadings(info, f"PC{i+1}")
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)

                    # Summary table
                    with st.expander("📋 Variance Summary Table"):
                        vdf = pd.DataFrame({
                            "Component": [f"PC{i+1}" for i in range(info["n_components"])],
                            "Variance Explained": [f"{v:.2%}" for v in info["explained_variance_ratio"]],
                            "Cumulative": [f"{v:.2%}" for v in info["cumulative_variance"]],
                        })
                        st.dataframe(vdf, use_container_width=True, hide_index=True)

                st.rerun()

    with ex2:
        if not dt_cols:
            st.info("No datetime columns detected.")
        else:
            st.markdown("Extracts: year, month, day, day-of-week, hour. Optionally adds sin/cos cyclical encoding.")
            dt_sel = st.multiselect("DateTime columns", dt_cols, default=dt_cols, key="fe_dt_c")
            dt_cyc = st.checkbox("Add cyclical encoding (sin/cos)", value=True, key="fe_dt_cyc")
            if st.button("▶ Extract DateTime Features", key="btn_dt", disabled=not dt_sel):
                log, _ = agent.apply_datetime_extraction(dt_sel, cyclical=dt_cyc)
                _ok(log.detail)
                st.rerun()

    # ══════════════════════════════════════════════
    # ④ FEATURE SELECTION
    # ══════════════════════════════════════════════
    _h("④ Feature Selection")
    st.caption("Choose the most relevant features for your problem. Removes noise, improves model performance.")

    # ── Target + Task config (shared by all selection methods) ──
    sc1, sc2 = st.columns(2)
    sel_target = sc1.selectbox("Target column", agent.columns, key="fe_sel_target")
    sel_task = sc2.selectbox("Task type", ["classification", "regression"], key="fe_sel_task")
    feat_cols = [c for c in num_cols if c != sel_target]

    if len(feat_cols) < 2:
        st.warning("Need at least 2 numeric feature columns (besides target) for selection.")
    else:
        # ══════════════════════════════════════════
        # 🚀 AUTO FEATURE SELECTOR (prominent)
        # ══════════════════════════════════════════
        st.markdown("""
        <div class="fe-auto">
            <h3>🎯 Auto Feature Selector</h3>
            <p>Runs <strong>5 methods simultaneously</strong> (Mutual Information, ANOVA/F-test, Correlation, 
            Random Forest Importance, Lasso L1) and aggregates rankings to find the consensus best features.</p>
        </div>
        """, unsafe_allow_html=True)

        auto_k = st.slider("Top K features to keep", 3,
                             min(50, len(feat_cols)), min(10, len(feat_cols)), key="fe_auto_k")

        if st.button("🚀 Run Auto Feature Selection", key="btn_auto",
                      type="primary", use_container_width=True):
            with st.spinner("Running 5 selection methods... This may take a moment."):
                log, info = agent.apply_auto_select(feat_cols, sel_target, sel_task, top_k=auto_k)

            _ok(log.detail)

            if info:
                # ── Chart: Multi-method ranking comparison ──
                rank_fig = _auto_rankings(info)
                if rank_fig:
                    st.plotly_chart(rank_fig, use_container_width=True)

                # ── Selected vs Dropped ──
                sel_l, sel_r = st.columns(2)
                with sel_l:
                    st.markdown("**✅ Selected Features**")
                    for f in info.get("selected", []):
                        r = info.get("average_ranks", {}).get(f, "?")
                        st.markdown(f"- `{f}` — avg rank: **{r:.1f}**" if isinstance(r, float) else f"- `{f}`")
                with sel_r:
                    st.markdown("**❌ Dropped Features**")
                    for f in info.get("dropped", []):
                        r = info.get("average_ranks", {}).get(f, "?")
                        st.markdown(f"- ~~`{f}`~~ — avg rank: {r:.1f}" if isinstance(r, float) else f"- ~~`{f}`~~")

                # ── Per-method detail tables ──
                with st.expander("📊 Detailed Rankings Per Method"):
                    rankings = info.get("method_rankings", {})
                    for method, mr in rankings.items():
                        st.markdown(f"**{method}**")
                        sorted_r = sorted(mr.items(), key=lambda x: x[1])
                        rdf = pd.DataFrame(sorted_r, columns=["Feature", "Rank"])
                        st.dataframe(rdf, use_container_width=True, hide_index=True, height=200)

            st.rerun()

        # ══════════════════════════════════════════
        # Individual Methods
        # ══════════════════════════════════════════
        st.markdown("---")
        st.markdown("**Or run individual selection methods:**")

        fs1, fs2, fs3 = st.tabs(["🔬 Filter Methods", "🔄 Wrapper (RFE)", "🌳 Embedded Methods"])

        # ── FILTER ──
        with fs1:
            ft = st.selectbox("Filter method", [
                "Variance Threshold", "Correlation Filter", "Mutual Information", "Statistical Test (ANOVA/F)"
            ], key="fe_ft")

            if ft == "Variance Threshold":
                vt = st.slider("Threshold", 0.0, 1.0, 0.01, 0.005, key="fe_vt")
                st.caption("Drops features with variance below this value. Low variance = feature is nearly constant.")
                if st.button("▶ Apply Variance Filter", key="btn_var"):
                    log, info = agent.apply_variance_threshold(feat_cols, threshold=vt)
                    _ok(log.detail)
                    if info and "variances" in info:
                        st.plotly_chart(_bar(info["variances"], "Feature Variances (red line = threshold)",
                                             "Variance", "#10b981"), use_container_width=True)
                    st.rerun()

            elif ft == "Correlation Filter":
                ct = st.slider("Threshold", 0.7, 1.0, 0.95, 0.01, key="fe_ct")
                st.caption("If two features correlate above this threshold, one is dropped.")
                if st.button("▶ Apply Correlation Filter", key="btn_corr"):
                    log, info = agent.apply_correlation_filter(feat_cols, threshold=ct)
                    _ok(log.detail)
                    if info and "correlation_matrix" in info:
                        st.plotly_chart(_heatmap(info["correlation_matrix"],
                                        f"Correlation Matrix (drop if > {ct})"), use_container_width=True)
                        if info.get("drop_reasons"):
                            with st.expander("📋 Why each feature was dropped"):
                                for col, reasons in info["drop_reasons"].items():
                                    partners = ", ".join(f"{k} ({v:.3f})" for k, v in reasons.items())
                                    st.markdown(f"- **{col}** → highly correlated with: {partners}")
                    st.rerun()

            elif ft == "Mutual Information":
                mi_k = st.slider("Top K", 3, max(3, len(feat_cols)), min(10, len(feat_cols)), key="fe_mik")
                st.caption("MI measures how much knowing a feature reduces uncertainty about the target.")
                if st.button("▶ Rank by Mutual Info", key="btn_mi"):
                    log, info = agent.apply_mutual_info(feat_cols, sel_target, sel_task, k=mi_k)
                    _ok(log.detail)
                    if info and "scores" in info:
                        st.plotly_chart(_bar(info["scores"], "Mutual Information Scores",
                                             "MI Score", "#f59e0b"), use_container_width=True)
                    st.rerun()

            elif ft == "Statistical Test (ANOVA/F)":
                sk = st.slider("Top K", 3, max(3, len(feat_cols)), min(10, len(feat_cols)), key="fe_sk")
                st.caption("ANOVA F-test (classification) or F-regression tests statistical significance.")
                if st.button("▶ Run Statistical Test", key="btn_stat"):
                    log, info = agent.apply_statistical_test(feat_cols, sel_target, sel_task, k=sk)
                    _ok(log.detail)
                    if info and "scores" in info:
                        fsc = {k: v["f_score"] for k, v in info["scores"].items()}
                        st.plotly_chart(_bar(fsc, f'{info.get("test","F-test")} Scores',
                                             "F-Score", "#ef4444"), use_container_width=True)
                        with st.expander("📋 P-values"):
                            pv = {k: v["p_value"] for k, v in info["scores"].items()}
                            pvdf = pd.DataFrame(sorted(pv.items(), key=lambda x: x[1]),
                                                 columns=["Feature", "P-Value"])
                            pvdf["Significant (p<0.05)"] = pvdf["P-Value"].apply(
                                lambda p: "✅ Yes" if p < 0.05 else "❌ No")
                            st.dataframe(pvdf, use_container_width=True, hide_index=True)
                    st.rerun()

        # ── WRAPPER ──
        with fs2:
            st.markdown("""
            **Recursive Feature Elimination (RFE)** trains a model, removes the least important feature, 
            retrains, and repeats until the desired number of features remains. Uses RandomForest internally.
            """)
            rfe_n = st.slider("Features to select", 3, max(3, len(feat_cols)),
                                min(10, len(feat_cols)), key="fe_rfen")
            if st.button("▶ Run RFE", key="btn_rfe"):
                with st.spinner("Running RFE (may take a moment)..."):
                    log, info = agent.apply_rfe(feat_cols, sel_target, sel_task, n_features=rfe_n)
                _ok(log.detail)
                if info and "rankings" in info:
                    st.plotly_chart(_bar(info["rankings"],
                                    "RFE Rankings (1 = selected, higher = eliminated earlier)",
                                    "Rank (1=best)", "#3b82f6"), use_container_width=True)
                    with st.expander("📋 Selected vs Eliminated"):
                        s_col, d_col = st.columns(2)
                        with s_col:
                            st.markdown("**✅ Selected**")
                            for f in info.get("selected", []):
                                st.markdown(f"- `{f}` (rank {info['rankings'].get(f, '?')})")
                        with d_col:
                            st.markdown("**❌ Eliminated**")
                            for f in info.get("dropped", []):
                                st.markdown(f"- ~~`{f}`~~ (rank {info['rankings'].get(f, '?')})")
                st.rerun()

        # ── EMBEDDED ──
        with fs3:
            emb = st.selectbox("Method", [
                "🌳 Tree-Based Importance (Random Forest)",
                "📏 Lasso (L1 Regularization)",
            ], key="fe_emb")

            if "Tree" in emb:
                st.markdown("""
                Trains a Random Forest and uses feature importances (Gini impurity) to rank features.
                Features below the threshold are dropped.
                """)
                tt = st.slider("Importance threshold", 0.001, 0.1, 0.01, 0.005, key="fe_tt")
                if st.button("▶ Run Tree Importance", key="btn_tree"):
                    with st.spinner("Training Random Forest..."):
                        log, info = agent.apply_tree_importance(feat_cols, sel_target, sel_task, threshold=tt)
                    _ok(log.detail)
                    if info and "importances" in info:
                        st.plotly_chart(_bar(info["importances"],
                                        f"RF Feature Importances (threshold={tt})",
                                        "Importance", "#10b981"), use_container_width=True)
                        with st.expander("📋 Selected vs Dropped"):
                            s_col, d_col = st.columns(2)
                            with s_col:
                                st.markdown("**✅ Kept** (importance ≥ threshold)")
                                for f in info.get("selected", []):
                                    st.markdown(f"- `{f}` ({info['importances'].get(f, 0):.4f})")
                            with d_col:
                                st.markdown("**❌ Dropped** (importance < threshold)")
                                for f in info.get("dropped", []):
                                    st.markdown(f"- ~~`{f}`~~ ({info['importances'].get(f, 0):.4f})")
                    st.rerun()

            elif "Lasso" in emb:
                st.markdown("""
                Lasso applies L1 regularization which shrinks unimportant feature coefficients to exactly zero.
                Features with zero coefficients are automatically removed.
                """)
                la = st.slider("Alpha (regularization strength)", 0.001, 1.0, 0.01, 0.005, key="fe_la")
                if st.button("▶ Run Lasso Selection", key="btn_lasso"):
                    log, info = agent.apply_lasso_selection(feat_cols, sel_target, sel_task, alpha=la)
                    _ok(log.detail)
                    if info and "coefficients" in info:
                        st.plotly_chart(_bar(info["coefficients"],
                                        f"Lasso |Coefficients| (α={la})",
                                        "|Coefficient|", "#f59e0b"), use_container_width=True)
                        if info.get("dropped"):
                            st.markdown(f"**Zero-coefficient (dropped):** {', '.join(f'`{d}`' for d in info['dropped'])}")
                    st.rerun()

    # ══════════════════════════════════════════════
    # ⑤ FEATURE SCALING
    # ══════════════════════════════════════════════
    _h("⑤ Feature Scaling")
    st.caption("Ensures all features contribute equally. Important for distance-based models (KNN, SVM, etc.).")

    if not num_cols:
        st.info("No numeric columns.")
    else:
        scl = st.selectbox("Method", [
            "Standard Scaling (z-score: mean=0, variance=1)",
            "Min-Max Scaling [0, 1]",
            "Robust Scaling (median/IQR — outlier resistant)",
            "MaxAbs Scaling [-1, 1]",
        ], key="fe_scl")
        scl_cols = st.multiselect("Columns", num_cols, key="fe_scl_c")

        if st.button("▶ Apply Scaling", key="btn_scl", disabled=not scl_cols):
            if "Standard" in scl:
                log, _ = agent.apply_standard_scaling(scl_cols)
            elif "Min-Max" in scl:
                log, _ = agent.apply_minmax_scaling(scl_cols)
            elif "Robust" in scl:
                log, _ = agent.apply_robust_scaling(scl_cols)
            elif "MaxAbs" in scl:
                log, _ = agent.apply_maxabs_scaling(scl_cols)
            _ok(log.detail)
            st.rerun()

    # ══════════════════════════════════════════════
    # ⑥ DROP COLUMNS
    # ══════════════════════════════════════════════
    _h("⑥ Drop Columns")
    drop_sel = st.multiselect("Select columns to remove", agent.columns, key="fe_drop")
    if st.button("▶ Drop Selected", key="btn_drop", disabled=not drop_sel):
        log, _ = agent.apply_drop_columns(drop_sel)
        _ok(log.detail)
        st.rerun()

    # ══════════════════════════════════════════════
    # ⑦ OPERATION LOG + UNDO / RESET
    # ══════════════════════════════════════════════
    _h("⑦ Operation Log")

    if not agent.logs:
        st.info("No transforms applied yet. Use the sections above to engineer features.")
    else:
        for i, log in enumerate(agent.logs, 1):
            cat = log.category if hasattr(log, "category") else "utility"
            st.markdown(f"""
            <div class="fe-log">
                <strong>#{i}</strong> — <strong>{log.operation}</strong> {_badge(cat)}<br>
                <span style="color:#94a3b8;">{log.detail}</span><br>
                <span style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#64748b;">
                    Cols: {log.cols_before} → {log.cols_after}
                </span>
            </div>
            """, unsafe_allow_html=True)

    u1, u2 = st.columns(2)
    with u1:
        if st.button("↩ Undo Last", key="btn_undo", disabled=not agent.logs, use_container_width=True):
            if agent.undo():
                st.toast("Undone!", icon="↩")
                st.rerun()
    with u2:
        if st.button("🔄 Reset All", key="btn_reset", disabled=not agent.logs,
                      use_container_width=True, type="secondary"):
            agent.reset()
            st.toast("Reset to original.", icon="🔄")
            st.rerun()

    # ══════════════════════════════════════════════
    # ⑧ PREVIEW + DOWNLOAD + SEND TO ML
    # ══════════════════════════════════════════════
    _h("⑧ Preview & Export")

    st.dataframe(agent.current_df.head(50), use_container_width=True, height=300)
    st.caption(f"Showing first 50 of {agent.shape[0]:,} rows × {agent.shape[1]} columns")

    d1, d2, d3 = st.columns(3)
    with d1:
        csv = agent.current_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download CSV", data=csv, file_name="engineered_data.csv",
                            mime="text/csv", use_container_width=True, key="fe_dl_csv")
    with d2:
        from utils.helpers import dataframe_to_excel_bytes
        xlsx = dataframe_to_excel_bytes(agent.current_df)
        st.download_button("⬇ Download Excel", data=xlsx, file_name="engineered_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True, key="fe_dl_xlsx")
    with d3:
        if st.button("🚀 Send to ML Training →", use_container_width=True, type="primary", key="fe_send"):
            st.session_state["fe_engineered_df"] = agent.current_df.copy()
            for key in ["ml_agent", "ml_analysis", "ml_recommendations", "ml_all_models",
                         "ml_single_result", "ml_cmp_result_a", "ml_cmp_result_b", "ml_comparison"]:
                st.session_state.pop(key, None)
            st.toast("Engineered data sent to ML tab!", icon="🚀")