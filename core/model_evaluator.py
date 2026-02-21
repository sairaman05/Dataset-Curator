# core/model_evaluator.py
"""
Evaluation charts and metric visualizations using Plotly.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
from typing import List, Optional, Dict

from core.model_trainer import TrainingResult, EpochMetric


# ─── Dark Theme ────────────────────────────────────────
DARK_BG = "#0a0e1a"
DARK_PAPER = "#0f1320"
DARK_GRID = "#1a1f35"
ACCENT = "#6366f1"
ACCENT2 = "#f472b6"
TEXT_COLOR = "#e2e8f0"
FONT_FAMILY = "DM Sans, sans-serif"

def _dark_layout() -> dict:
    """Base dark theme layout properties."""
    return dict(
        paper_bgcolor=DARK_PAPER,
        plot_bgcolor=DARK_BG,
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR, size=12),
        xaxis=dict(gridcolor=DARK_GRID, zerolinecolor=DARK_GRID),
        yaxis=dict(gridcolor=DARK_GRID, zerolinecolor=DARK_GRID),
        margin=dict(l=60, r=30, t=50, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR)),
    )


def training_progress_chart(history: List[EpochMetric], title: str = "Training Progress") -> go.Figure:
    """Line chart of train/val score and loss over epochs."""
    epochs = [e.epoch for e in history]
    train_scores = [e.train_score for e in history]
    val_scores = [e.val_score for e in history]
    train_losses = [e.train_loss for e in history]
    val_losses = [e.val_loss for e in history]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Score (↑ better)", "Loss (↓ better)"))

    fig.add_trace(go.Scatter(x=epochs, y=train_scores, name="Train Score",
                             line=dict(color=ACCENT, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=val_scores, name="Val Score",
                             line=dict(color=ACCENT2, width=2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=train_losses, name="Train Loss",
                             line=dict(color=ACCENT, width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=val_losses, name="Val Loss",
                             line=dict(color=ACCENT2, width=2, dash="dash")), row=1, col=2)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR)),
        height=400,
        **_dark_layout()
    )
    fig.update_xaxes(title_text="Epoch", gridcolor=DARK_GRID)
    fig.update_yaxes(gridcolor=DARK_GRID)
    return fig


def live_training_chart(history: List[EpochMetric]) -> go.Figure:
    """Build a live-updating chart for epoch-by-epoch display."""
    epochs = [e.epoch for e in history]
    train_losses = [e.train_loss for e in history]
    val_losses = [e.val_loss for e in history]
    train_scores = [e.train_score for e in history]
    val_scores = [e.val_score for e in history]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Loss Curve (↓ better)", "Score Curve (↑ better)")
    )

    fig.add_trace(go.Scatter(x=epochs, y=train_losses, name="Train Loss",
                             line=dict(color="#f97316", width=2.5), mode="lines+markers",
                             marker=dict(size=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=val_losses, name="Val Loss",
                             line=dict(color="#ef4444", width=2.5, dash="dash"), mode="lines+markers",
                             marker=dict(size=4)), row=1, col=1)

    fig.add_trace(go.Scatter(x=epochs, y=train_scores, name="Train Score",
                             line=dict(color="#22c55e", width=2.5), mode="lines+markers",
                             marker=dict(size=4)), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=val_scores, name="Val Score",
                             line=dict(color="#3b82f6", width=2.5, dash="dash"), mode="lines+markers",
                             marker=dict(size=4)), row=1, col=2)

    fig.update_layout(
        title=dict(text="📊 Live Training Progress", font=dict(size=16, color=TEXT_COLOR)),
        height=380,
        showlegend=True,
        **_dark_layout()
    )
    fig.update_xaxes(title_text="Epoch", gridcolor=DARK_GRID)
    fig.update_yaxes(gridcolor=DARK_GRID)
    return fig


def confusion_matrix_chart(y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: Optional[List[str]] = None) -> go.Figure:
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = class_names or [str(i) for i in range(cm.shape[0])]

    # Normalize for color but show raw counts as text
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    text = [[f"{cm[i][j]}<br>({cm_norm[i][j]:.1%})" for j in range(cm.shape[1])] for i in range(cm.shape[0])]

    fig = go.Figure(data=go.Heatmap(
        z=cm_norm, x=labels, y=labels,
        text=text, texttemplate="%{text}",
        colorscale=[[0, "#0f1320"], [0.5, "#4338ca"], [1, "#6366f1"]],
        showscale=True,
        colorbar=dict(title="Ratio", tickfont=dict(color=TEXT_COLOR))
    ))

    fig.update_layout(
        title=dict(text="Confusion Matrix", font=dict(size=16, color=TEXT_COLOR)),
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=450,
        **_dark_layout()
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def feature_importance_chart(importances: np.ndarray, feature_names: List[str],
                              top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of top feature importances."""
    idx = np.argsort(importances)[-top_n:]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=vals, colorscale=[[0, "#312e81"], [1, "#6366f1"]]),
        text=[f"{v:.4f}" for v in vals], textposition="outside",
        textfont=dict(color=TEXT_COLOR, size=10),
    ))
    fig.update_layout(
        title=dict(text=f"Top {min(top_n, len(feature_names))} Feature Importances",
                   font=dict(size=16, color=TEXT_COLOR)),
        height=max(350, 30 * top_n),
        **_dark_layout()
    )
    return fig


def actual_vs_predicted_chart(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Scatter plot of actual vs predicted values for regression."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers",
        marker=dict(color=ACCENT, size=6, opacity=0.6),
        name="Predictions"
    ))

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", line=dict(color=ACCENT2, width=2, dash="dash"),
        name="Perfect Prediction"
    ))

    fig.update_layout(
        title=dict(text="Actual vs Predicted", font=dict(size=16, color=TEXT_COLOR)),
        xaxis_title="Actual", yaxis_title="Predicted",
        height=450,
        **_dark_layout()
    )
    return fig


def residual_plot(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Residuals plot for regression."""
    residuals = y_true - y_pred

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Residuals vs Predicted", "Residual Distribution"))

    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals, mode="markers",
        marker=dict(color=ACCENT, size=5, opacity=0.6),
        name="Residuals"
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=ACCENT2, row=1, col=1)

    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=40,
        marker=dict(color=ACCENT, line=dict(color=TEXT_COLOR, width=0.5)),
        name="Distribution"
    ), row=1, col=2)

    fig.update_layout(
        title=dict(text="Residual Analysis", font=dict(size=16, color=TEXT_COLOR)),
        height=400,
        **_dark_layout()
    )
    fig.update_xaxes(gridcolor=DARK_GRID)
    fig.update_yaxes(gridcolor=DARK_GRID)
    return fig


def metrics_comparison_chart(results: List[TrainingResult]) -> go.Figure:
    """Bar chart comparing metrics across multiple models."""
    if not results:
        return go.Figure()

    model_names = [r.display_name for r in results]
    is_clf = "classification" in results[0].task_type

    if is_clf:
        metric_keys = ["accuracy", "precision", "recall", "f1_score"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    else:
        metric_keys = ["r2", "rmse", "mae"]
        metric_labels = ["R²", "RMSE", "MAE"]

    colors = ["#6366f1", "#f472b6", "#22c55e", "#f97316", "#3b82f6"]

    fig = go.Figure()
    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        values = [r.metrics.get(key, 0) for r in results]
        fig.add_trace(go.Bar(
            name=label, x=model_names, y=values,
            marker_color=colors[i % len(colors)],
            text=[f"{v:.4f}" for v in values], textposition="outside",
            textfont=dict(size=10, color=TEXT_COLOR),
        ))

    fig.update_layout(
        title=dict(text="Model Comparison — Evaluation Metrics", font=dict(size=16, color=TEXT_COLOR)),
        barmode="group",
        height=450,
        **_dark_layout()
    )
    return fig


def epoch_comparison_chart(results: List[TrainingResult]) -> go.Figure:
    """Overlay training curves from multiple models."""
    if not results:
        return go.Figure()

    colors = ["#6366f1", "#f472b6", "#22c55e", "#f97316", "#3b82f6", "#a78bfa"]
    fig = go.Figure()

    for i, r in enumerate(results):
        color = colors[i % len(colors)]
        epochs = [e.epoch for e in r.epoch_history]
        val_scores = [e.val_score for e in r.epoch_history]
        val_losses = [e.val_loss for e in r.epoch_history]

        fig.add_trace(go.Scatter(
            x=epochs, y=val_scores, name=f"{r.display_name} (Val Score)",
            line=dict(color=color, width=2), mode="lines"
        ))

    fig.update_layout(
        title=dict(text="Validation Score Comparison Across Epochs", font=dict(size=16, color=TEXT_COLOR)),
        xaxis_title="Epoch", yaxis_title="Validation Score",
        height=400,
        **_dark_layout()
    )
    return fig