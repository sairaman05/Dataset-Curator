"""Auto-generate visualizations for cleaned data."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config.settings import CHART_HEIGHT


class DataVisualizer:
    """Generate Plotly charts for data exploration."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def null_comparison_chart(self, null_comparison_df: pd.DataFrame) -> go.Figure:
        """Bar chart comparing nulls before and after cleaning."""
        if null_comparison_df.empty:
            return None

        # Only show columns that had nulls
        has_nulls = null_comparison_df[null_comparison_df["Nulls Before"] > 0]
        if has_nulls.empty:
            return None

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Before Cleaning",
            x=has_nulls["Column"],
            y=has_nulls["Nulls Before"],
            marker_color="#EF4444",
        ))
        fig.add_trace(go.Bar(
            name="After Cleaning",
            x=has_nulls["Column"],
            y=has_nulls["Nulls After"],
            marker_color="#22C55E",
        ))
        fig.update_layout(
            title="Null Values: Before vs After Cleaning",
            barmode="group",
            height=CHART_HEIGHT,
            template="plotly_white",
        )
        return fig

    def dtype_distribution_chart(self) -> go.Figure:
        """Pie chart of column dtype distribution."""
        dtype_counts = self.df.dtypes.astype(str).value_counts()
        fig = px.pie(
            names=dtype_counts.index,
            values=dtype_counts.values,
            title="Column Data Type Distribution",
            hole=0.4,
        )
        fig.update_layout(height=CHART_HEIGHT, template="plotly_white")
        return fig

    def numeric_distributions(self, max_cols: int = 9) -> go.Figure | None:
        """Histogram grid for numeric columns."""
        numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            return None

        cols_to_plot = numeric_cols[:max_cols]
        n = len(cols_to_plot)
        n_rows = (n + 2) // 3
        n_chart_cols = min(n, 3)

        fig = make_subplots(
            rows=n_rows,
            cols=n_chart_cols,
            subplot_titles=cols_to_plot,
        )

        for i, col in enumerate(cols_to_plot):
            row = i // n_chart_cols + 1
            col_idx = i % n_chart_cols + 1
            fig.add_trace(
                go.Histogram(x=self.df[col], name=col, showlegend=False,
                             marker_color="#6366F1"),
                row=row, col=col_idx,
            )

        fig.update_layout(
            title="Numeric Column Distributions",
            height=CHART_HEIGHT * n_rows * 0.7,
            template="plotly_white",
            showlegend=False,
        )
        return fig

    def categorical_distributions(self, max_cols: int = 6) -> go.Figure | None:
        """Bar charts for categorical columns (top 10 values each)."""
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not cat_cols:
            return None

        cols_to_plot = cat_cols[:max_cols]
        n = len(cols_to_plot)
        n_rows = (n + 1) // 2
        n_chart_cols = min(n, 2)

        fig = make_subplots(
            rows=n_rows,
            cols=n_chart_cols,
            subplot_titles=cols_to_plot,
        )

        for i, col in enumerate(cols_to_plot):
            row = i // n_chart_cols + 1
            col_idx = i % n_chart_cols + 1
            top_vals = self.df[col].value_counts().head(10)
            fig.add_trace(
                go.Bar(x=top_vals.index.astype(str), y=top_vals.values,
                       name=col, showlegend=False, marker_color="#F59E0B"),
                row=row, col=col_idx,
            )

        fig.update_layout(
            title="Categorical Column Distributions (Top 10 Values)",
            height=CHART_HEIGHT * n_rows * 0.7,
            template="plotly_white",
            showlegend=False,
        )
        return fig

    def correlation_heatmap(self) -> go.Figure | None:
        """Correlation heatmap for numeric columns."""
        numeric_df = self.df.select_dtypes(include=["number"])
        if numeric_df.shape[1] < 2:
            return None

        corr = numeric_df.corr().round(3)
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Correlation Heatmap",
        )
        fig.update_layout(height=CHART_HEIGHT + 100, template="plotly_white")
        return fig