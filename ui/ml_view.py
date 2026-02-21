# ui/ml_view.py
"""
Complete ML Tab UI — Redesigned.

Layout:
  ① Target selector + Analyze button
  ② Data analysis results + Top recommendations
  ③ SINGLE MODEL TRAINING — select model, epochs, train, live chart, results, download report+model
  ④ COMPARISON TRAINING — select 2 models, epochs, train both, live charts, comparison, download
"""

import streamlit as st
import pickle
import time
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Optional

from agents.ml_agent import MLAgent
from core.model_registry import get_registry
from core.model_evaluator import (
    training_progress_chart, live_training_chart, confusion_matrix_chart,
    feature_importance_chart, actual_vs_predicted_chart, residual_plot,
    metrics_comparison_chart, epoch_comparison_chart,
)
from config.settings import MIN_EPOCHS, MAX_EPOCHS, DEFAULT_EPOCHS


# ─── CSS ───────────────────────────────────────────────
ML_CSS = """
<style>
.ml-section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #e2e8f0;
    padding: 0.5rem 0;
    border-bottom: 2px solid #6366f1;
    margin: 1.5rem 0 0.8rem 0;
}
.ml-subsection {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #c7d2fe;
    margin: 1rem 0 0.5rem 0;
}
.score-badge {
    display: inline-block;
    background: linear-gradient(135deg, #4338ca, #6366f1);
    color: white;
    padding: 0.15rem 0.6rem;
    border-radius: 16px;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card-sm {
    background: linear-gradient(135deg, #0f1320, #1a1f35);
    border: 1px solid #2d3555;
    border-radius: 10px;
    padding: 0.8rem;
    text-align: center;
}
.metric-card-sm .label { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.04em; }
.metric-card-sm .value { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; color: #c7d2fe; }
.winner-box {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
}
.cplx-low { background: #064e3b; color: #6ee7b7; padding: 0.1rem 0.5rem; border-radius: 10px; font-size: 0.7rem; font-weight: 600; }
.cplx-med { background: #713f12; color: #fbbf24; padding: 0.1rem 0.5rem; border-radius: 10px; font-size: 0.7rem; font-weight: 600; }
.cplx-high { background: #7f1d1d; color: #fca5a5; padding: 0.1rem 0.5rem; border-radius: 10px; font-size: 0.7rem; font-weight: 600; }
</style>
"""


def _section(text: str):
    st.markdown(f'<div class="ml-section-header">{text}</div>', unsafe_allow_html=True)


def _subsection(text: str):
    st.markdown(f'<div class="ml-subsection">{text}</div>', unsafe_allow_html=True)


def _metric_sm(col, label: str, value: str):
    col.markdown(f'<div class="metric-card-sm"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)


def _cplx_tag(complexity: str) -> str:
    c = complexity.lower().replace("-", "")
    if "low" in c:
        return '<span class="cplx-low">Low</span>'
    if "high" in c:
        return '<span class="cplx-high">High</span>'
    return '<span class="cplx-med">Medium</span>'


def _model_to_bytes(model) -> bytes:
    """Serialize a trained model to bytes for download."""
    buf = BytesIO()
    pickle.dump(model, buf)
    return buf.getvalue()


def _run_training(agent, model_name, cleaned_df, analysis, epochs, prefix="single"):
    """
    Shared training logic with live chart display.
    Returns: TrainingResult or None on failure.
    """
    _subsection("⏳ Training in Progress...")

    progress_bar = st.progress(0, text="Initializing...")
    chart_placeholder = st.empty()

    mc1, mc2, mc3, mc4 = st.columns(4)
    epoch_ph = mc1.empty()
    train_ph = mc2.empty()
    val_ph = mc3.empty()
    time_ph = mc4.empty()

    live_history = []

    def progress_callback(step, total, epoch_metric):
        live_history.append(epoch_metric)
        pct = step / total
        progress_bar.progress(pct, text=f"Epoch {step}/{total}")

        epoch_ph.markdown(f'<div class="metric-card-sm"><div class="label">Epoch</div><div class="value">{step}/{total}</div></div>', unsafe_allow_html=True)
        train_ph.markdown(f'<div class="metric-card-sm"><div class="label">Train Score</div><div class="value">{epoch_metric.train_score:.4f}</div></div>', unsafe_allow_html=True)
        val_ph.markdown(f'<div class="metric-card-sm"><div class="label">Val Score</div><div class="value">{epoch_metric.val_score:.4f}</div></div>', unsafe_allow_html=True)
        time_ph.markdown(f'<div class="metric-card-sm"><div class="label">Time</div><div class="value">{epoch_metric.elapsed_time:.1f}s</div></div>', unsafe_allow_html=True)

        update_interval = max(1, total // 100)
        if step % update_interval == 0 or step == total:
            fig = live_training_chart(live_history)
            chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"{prefix}_live_{step}")

    try:
        result = agent.train(
            model_class_name=model_name,
            df=cleaned_df,
            target_column=analysis.target_column,
            analysis=analysis,
            epochs=epochs,
            progress_callback=progress_callback,
        )
        progress_bar.progress(1.0, text="✅ Training Complete!")
        st.success(f"✅ **{result.display_name}** trained in {result.total_time:.2f}s")
        return result
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        import traceback
        with st.expander("Full traceback"):
            st.code(traceback.format_exc())
        return None


def _render_results(result, prefix="single"):
    """Shared result rendering: metrics, charts, epoch history."""
    is_clf = "classification" in result.task_type

    _subsection("📊 Evaluation Metrics")

    if is_clf:
        c1, c2, c3, c4 = st.columns(4)
        _metric_sm(c1, "Accuracy", f"{result.metrics.get('accuracy', 0):.4f}")
        _metric_sm(c2, "Precision", f"{result.metrics.get('precision', 0):.4f}")
        _metric_sm(c3, "Recall", f"{result.metrics.get('recall', 0):.4f}")
        _metric_sm(c4, "F1 Score", f"{result.metrics.get('f1_score', 0):.4f}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        _metric_sm(c1, "R²", f"{result.metrics.get('r2', 0):.4f}")
        _metric_sm(c2, "RMSE", f"{result.metrics.get('rmse', 0):.4f}")
        _metric_sm(c3, "MAE", f"{result.metrics.get('mae', 0):.4f}")
        mape = result.metrics.get('mape')
        _metric_sm(c4, "MAPE", f"{mape:.2f}%" if mape else "N/A")

    st.markdown("")

    # Training progress
    st.plotly_chart(
        training_progress_chart(result.epoch_history, f"{result.display_name} — Training Progress"),
        use_container_width=True, key=f"{prefix}_progress"
    )

    # Confusion Matrix or Actual vs Predicted + Feature Importance
    col1, col2 = st.columns(2)
    with col1:
        if is_clf:
            st.plotly_chart(
                confusion_matrix_chart(result.test_true, result.test_predictions, result.class_names),
                use_container_width=True, key=f"{prefix}_cm"
            )
        else:
            st.plotly_chart(
                actual_vs_predicted_chart(result.test_true, result.test_predictions),
                use_container_width=True, key=f"{prefix}_avp"
            )
    with col2:
        if result.feature_importances is not None:
            st.plotly_chart(
                feature_importance_chart(result.feature_importances, result.feature_names),
                use_container_width=True, key=f"{prefix}_fi"
            )
        elif not is_clf:
            st.plotly_chart(
                residual_plot(result.test_true, result.test_predictions),
                use_container_width=True, key=f"{prefix}_resid"
            )

    if not is_clf and result.feature_importances is not None:
        st.plotly_chart(
            residual_plot(result.test_true, result.test_predictions),
            use_container_width=True, key=f"{prefix}_resid2"
        )

    # Full metrics table
    with st.expander("📋 Complete Metrics Table"):
        metrics_df = pd.DataFrame([
            {"Metric": k.replace("_", " ").title(), "Value": f"{v:.6f}"}
            for k, v in result.metrics.items()
        ])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Epoch history
    with st.expander("📊 Full Epoch History"):
        hist_data = [{
            "Epoch": e.epoch,
            "Train Score": f"{e.train_score:.6f}",
            "Val Score": f"{e.val_score:.6f}",
            "Train Loss": f"{e.train_loss:.6f}",
            "Val Loss": f"{e.val_loss:.6f}",
            "Time (s)": f"{e.elapsed_time:.2f}",
        } for e in result.epoch_history]
        st.dataframe(pd.DataFrame(hist_data), use_container_width=True, hide_index=True)


def _render_downloads(agent, result, prefix="single"):
    """Render download buttons for report and model."""
    _subsection("📥 Downloads")
    col1, col2 = st.columns(2)
    with col1:
        report_md = agent.generate_report(result)
        st.download_button(
            label=f"📄 Download Report (.md)",
            data=report_md,
            file_name=f"report_{result.model_name}_{result.total_epochs}ep.md",
            mime="text/markdown",
            use_container_width=True,
            key=f"{prefix}_dl_report",
        )
    with col2:
        model_bytes = _model_to_bytes(result.trained_model)
        st.download_button(
            label=f"💾 Download Model (.pkl)",
            data=model_bytes,
            file_name=f"model_{result.model_name}_{result.total_epochs}ep.pkl",
            mime="application/octet-stream",
            use_container_width=True,
            key=f"{prefix}_dl_model",
        )


# ═══════════════════════════════════════════════════════
# MAIN RENDER FUNCTION
# ═══════════════════════════════════════════════════════

def render_ml_tab(cleaned_df: Optional[pd.DataFrame]):
    """Render the complete ML tab."""
    st.markdown(ML_CSS, unsafe_allow_html=True)

    if cleaned_df is None:
        st.info("⬆️ Upload and clean a dataset in the **Clean & Profile** tab first.")
        return

    # ─── Initialize ─────────────────────────────────────
    if "ml_agent" not in st.session_state:
        st.session_state.ml_agent = MLAgent()
    agent: MLAgent = st.session_state.ml_agent

    # ════════════════════════════════════════════════════
    # SECTION 1: Target Selection + Analyze
    # ════════════════════════════════════════════════════
    _section("① Select Target Column")

    col1, col2 = st.columns([3, 1])
    with col1:
        target_col = st.selectbox(
            "Which column do you want to predict?",
            options=list(cleaned_df.columns),
            key="ml_target_select",
            help="This will be the target/label column. All other columns become features."
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_clicked = st.button("🔍 Analyze & Recommend", use_container_width=True, type="primary")

    # ════════════════════════════════════════════════════
    # SECTION 2: Analysis Results
    # ════════════════════════════════════════════════════
    if analyze_clicked:
        with st.spinner("Analyzing dataset..."):
            try:
                analysis = agent.analyze(cleaned_df, target_col)
                st.session_state.ml_analysis = analysis

                recommendations = agent.recommend(analysis, top_n=5)
                st.session_state.ml_recommendations = recommendations

                all_models = agent.get_all_models(analysis)
                st.session_state.ml_all_models = all_models

                # Clear previous results
                for key in ["ml_single_result", "ml_cmp_result_a", "ml_cmp_result_b", "ml_comparison"]:
                    st.session_state.pop(key, None)
                agent.clear_cache()

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                return

    if "ml_analysis" not in st.session_state:
        st.info("👆 Select a target column and click **Analyze & Recommend** to start.")
        return

    analysis = st.session_state.ml_analysis
    recommendations = st.session_state.ml_recommendations
    all_models = st.session_state.ml_all_models

    # Show analysis
    _section("② Data Analysis")

    c1, c2, c3, c4 = st.columns(4)
    _metric_sm(c1, "Task Type", analysis.task_type.replace("_", " ").title())
    _metric_sm(c2, "Samples", f"{analysis.n_samples:,}")
    _metric_sm(c3, "Features", str(analysis.n_features))
    _metric_sm(c4, "Classes" if analysis.n_classes > 0 else "Target Unique",
               str(analysis.n_classes) if analysis.n_classes > 0 else "Continuous")

    st.markdown("")
    c1, c2, c3, c4 = st.columns(4)
    _metric_sm(c1, "Numeric Features", str(analysis.n_numeric))
    _metric_sm(c2, "Categorical Features", str(analysis.n_categorical))
    _metric_sm(c3, "Outlier %", f"{analysis.outlier_pct:.1f}%")
    _metric_sm(c4, "Dataset Size", analysis.dataset_size_category.title())

    if analysis.has_imbalance:
        st.warning("⚠️ Class imbalance detected in the target column.")
    if analysis.has_high_dimensionality:
        st.info("ℹ️ High-dimensional dataset (>50 features).")

    # Recommendations
    _section("③ Model Recommendations")
    st.markdown(f"**Top 5 recommended** out of **{len(all_models)} available models**:")

    medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    for i, rec in enumerate(recommendations):
        with st.expander(f"{medal[i]} #{rec.rank} — {rec.model_info.display_name}  (Score: {rec.score})", expanded=(i == 0)):
            col1, col2, col3 = st.columns([2, 1, 1])
            col1.markdown(f"**{rec.model_info.description}**")
            col2.markdown(f'Score: <span class="score-badge">{rec.score}</span>', unsafe_allow_html=True)
            col3.markdown(f'Complexity: {_cplx_tag(rec.model_info.complexity)}', unsafe_allow_html=True)

            if rec.reasons:
                st.markdown("**✅ Why this model:**")
                for r in rec.reasons:
                    st.markdown(f"- {r}")
            if rec.warnings:
                st.markdown("**⚠️ Considerations:**")
                for w in rec.warnings:
                    st.markdown(f"- {w}")
            st.caption(f"Architecture: {rec.model_info.architecture}")

    # Build model selection options (shared between sections)
    model_display_map = {}
    for rec in all_models:
        label = f"{rec.model_info.display_name}  (Score: {rec.score}, {rec.model_info.complexity})"
        model_display_map[label] = rec.model_info.class_name
    model_labels = list(model_display_map.keys())

    # ════════════════════════════════════════════════════
    # SECTION 4: SINGLE MODEL TRAINING
    # ════════════════════════════════════════════════════
    _section("④ Single Model Training")

    s_col1, s_col2 = st.columns([3, 1])
    with s_col1:
        single_label = st.selectbox(
            "Select model", options=model_labels,
            key="single_model_select",
            help=f"All {len(all_models)} models available."
        )
    single_model_name = model_display_map[single_label]

    with s_col2:
        single_epochs = st.slider("Epochs", min_value=MIN_EPOCHS, max_value=MAX_EPOCHS,
                                   value=DEFAULT_EPOCHS, step=5, key="single_epochs")

    single_train_clicked = st.button("🚀 Train Single Model", use_container_width=True, type="primary", key="btn_single_train")

    if single_train_clicked:
        result = _run_training(agent, single_model_name, cleaned_df, analysis, single_epochs, prefix="single")
        if result:
            st.session_state.ml_single_result = result

    if "ml_single_result" in st.session_state:
        result = st.session_state.ml_single_result
        _render_results(result, prefix="single")
        _render_downloads(agent, result, prefix="single")

    # ════════════════════════════════════════════════════
    # SECTION 5: COMPARISON TRAINING
    # ════════════════════════════════════════════════════
    _section("⑤ Model Comparison Training")
    st.markdown("Select **two models** to train and compare side-by-side.")

    cc1, cc2, cc3 = st.columns([2, 2, 1])
    with cc1:
        cmp_label_a = st.selectbox("Model A", options=model_labels, key="cmp_model_a")
    with cc2:
        # Default to second option if available
        default_b = 1 if len(model_labels) > 1 else 0
        cmp_label_b = st.selectbox("Model B", options=model_labels, index=default_b, key="cmp_model_b")
    with cc3:
        cmp_epochs = st.slider("Epochs", min_value=MIN_EPOCHS, max_value=MAX_EPOCHS,
                                value=DEFAULT_EPOCHS, step=5, key="cmp_epochs")

    cmp_model_a = model_display_map[cmp_label_a]
    cmp_model_b = model_display_map[cmp_label_b]

    if cmp_model_a == cmp_model_b:
        st.warning("⚠️ Please select two different models to compare.")

    cmp_train_clicked = st.button(
        "⚖️ Train & Compare Both Models", use_container_width=True, type="primary", key="btn_cmp_train",
        disabled=(cmp_model_a == cmp_model_b)
    )

    if cmp_train_clicked and cmp_model_a != cmp_model_b:
        # Train Model A
        st.markdown(f"### 🅰️ Training: {cmp_label_a.split('(')[0].strip()}")
        result_a = _run_training(agent, cmp_model_a, cleaned_df, analysis, cmp_epochs, prefix="cmp_a")

        if result_a:
            st.session_state.ml_cmp_result_a = result_a

            # Train Model B
            st.markdown(f"### 🅱️ Training: {cmp_label_b.split('(')[0].strip()}")
            agent.clear_cache()  # Force data re-preparation for clean state
            result_b = _run_training(agent, cmp_model_b, cleaned_df, analysis, cmp_epochs, prefix="cmp_b")

            if result_b:
                st.session_state.ml_cmp_result_b = result_b

                # Compare
                comparison = agent.compare([result_a, result_b])
                st.session_state.ml_comparison = comparison

    # Show comparison results if available
    if "ml_comparison" in st.session_state:
        comparison = st.session_state.ml_comparison
        result_a = st.session_state.ml_cmp_result_a
        result_b = st.session_state.ml_cmp_result_b

        _section("⑥ Comparison Results")

        # Winner
        st.markdown(f"""
        <div class="winner-box">
            <span style="font-size:1.5rem;">🏆</span>
            <span style="font-family:'Playfair Display',serif; font-size:1.2rem; font-weight:700; color:#6ee7b7; margin-left:0.5rem;">
                {comparison.winner}
            </span>
            <br>
            <span style="font-size:0.85rem; color:#a7f3d0;">{comparison.winner_reason}</span>
        </div>
        """, unsafe_allow_html=True)

        # Side-by-side metrics
        st.plotly_chart(metrics_comparison_chart([result_a, result_b]), use_container_width=True, key="cmp_metrics")

        # Epoch overlay
        st.plotly_chart(epoch_comparison_chart([result_a, result_b]), use_container_width=True, key="cmp_epochs_chart")

        # Detailed table
        _subsection("📋 Detailed Metrics")
        table_data = []
        for metric, values in comparison.metric_comparison.items():
            row = {"Metric": metric.replace("_", " ").title()}
            for name, val in values.items():
                row[name] = f"{val:.6f}" if val is not None else "N/A"
            table_data.append(row)
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # Training time
        _subsection("⏱️ Training Time")
        tc1, tc2 = st.columns(2)
        _metric_sm(tc1, result_a.display_name, f"{result_a.total_time:.2f}s")
        _metric_sm(tc2, result_b.display_name, f"{result_b.total_time:.2f}s")

        # Side-by-side confusion matrices / charts
        is_clf = "classification" in result_a.task_type
        if is_clf:
            _subsection("🔢 Confusion Matrices")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{result_a.display_name}**")
                st.plotly_chart(confusion_matrix_chart(result_a.test_true, result_a.test_predictions, result_a.class_names),
                                use_container_width=True, key="cmp_cm_a")
            with col2:
                st.markdown(f"**{result_b.display_name}**")
                st.plotly_chart(confusion_matrix_chart(result_b.test_true, result_b.test_predictions, result_b.class_names),
                                use_container_width=True, key="cmp_cm_b")

        # Individual epoch histories
        with st.expander(f"📊 {result_a.display_name} — Epoch History"):
            hist_a = [{
                "Epoch": e.epoch, "Train Score": f"{e.train_score:.6f}",
                "Val Score": f"{e.val_score:.6f}", "Train Loss": f"{e.train_loss:.6f}",
                "Val Loss": f"{e.val_loss:.6f}", "Time": f"{e.elapsed_time:.2f}s",
            } for e in result_a.epoch_history]
            st.dataframe(pd.DataFrame(hist_a), use_container_width=True, hide_index=True)

        with st.expander(f"📊 {result_b.display_name} — Epoch History"):
            hist_b = [{
                "Epoch": e.epoch, "Train Score": f"{e.train_score:.6f}",
                "Val Score": f"{e.val_score:.6f}", "Train Loss": f"{e.train_loss:.6f}",
                "Val Loss": f"{e.val_loss:.6f}", "Time": f"{e.elapsed_time:.2f}s",
            } for e in result_b.epoch_history]
            st.dataframe(pd.DataFrame(hist_b), use_container_width=True, hide_index=True)

        # Downloads
        _subsection("📥 Downloads")

        # Comparison report
        cmp_report = agent.generate_comparison_report(comparison)
        st.download_button(
            label="📄 Download Comparison Report (.md)",
            data=cmp_report,
            file_name=f"comparison_{result_a.model_name}_vs_{result_b.model_name}.md",
            mime="text/markdown",
            use_container_width=True,
            key="cmp_dl_report",
        )

        # Individual reports + models
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{result_a.display_name}:**")
            report_a = agent.generate_report(result_a)
            st.download_button(
                label="📄 Report (.md)", data=report_a,
                file_name=f"report_{result_a.model_name}_{result_a.total_epochs}ep.md",
                mime="text/markdown", use_container_width=True, key="cmp_dl_report_a",
            )
            st.download_button(
                label="💾 Model (.pkl)", data=_model_to_bytes(result_a.trained_model),
                file_name=f"model_{result_a.model_name}.pkl",
                mime="application/octet-stream", use_container_width=True, key="cmp_dl_model_a",
            )
        with col2:
            st.markdown(f"**{result_b.display_name}:**")
            report_b = agent.generate_report(result_b)
            st.download_button(
                label="📄 Report (.md)", data=report_b,
                file_name=f"report_{result_b.model_name}_{result_b.total_epochs}ep.md",
                mime="text/markdown", use_container_width=True, key="cmp_dl_report_b",
            )
            st.download_button(
                label="💾 Model (.pkl)", data=_model_to_bytes(result_b.trained_model),
                file_name=f"model_{result_b.model_name}.pkl",
                mime="application/octet-stream", use_container_width=True, key="cmp_dl_model_b",
            )