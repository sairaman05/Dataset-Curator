# core/feature_engineer.py
"""
Feature Engineering Engine — Complete Implementation.

5 Major Categories:
  1. Feature Creation     — Domain patterns, synthetic combos, aggregation
  2. Feature Transformation — Encoding (label, one-hot, target, freq), math transforms (log, sqrt, boxcox)
  3. Feature Extraction   — PCA, aggregation/combination, datetime extraction
  4. Feature Selection    — Filter (variance, correlation, MI, chi2, ANOVA), Wrapper (RFE), Embedded (Lasso, Tree)
  5. Feature Scaling      — Standard, MinMax, Robust, MaxAbs

Every function: takes DataFrame → returns (DataFrame, FELog) or (DataFrame, extras, FELog)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    PolynomialFeatures, PowerTransformer,
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, chi2, f_classif, f_regression,
    mutual_info_classif, mutual_info_regression, RFE,
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from config.settings import (
    POLY_MAX_DEGREE, POLY_MAX_FEATURES,
    HIGH_CARDINALITY_ENCODE_TOP_N,
    VARIANCE_THRESHOLD_DEFAULT,
    CORRELATION_DROP_THRESHOLD,
    MAX_BINS_DISCRETIZE,
    RANDOM_STATE,
)


@dataclass
class FELog:
    """Single feature engineering operation log."""
    operation: str
    category: str           # "creation" | "transformation" | "extraction" | "selection" | "scaling"
    columns_in: List[str]
    columns_out: List[str]
    detail: str
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    intermediates: Optional[Dict[str, Any]] = None   # For visualizations / explanations


@dataclass
class FEResult:
    original_df: pd.DataFrame
    engineered_df: pd.DataFrame
    logs: List[FELog]
    summary: Dict[str, int]


# ═══════════════════════════════════════════════════════
# 1. FEATURE CREATION
# ═══════════════════════════════════════════════════════

def create_polynomial(df: pd.DataFrame, columns: List[str],
                       degree: int = 2, interaction_only: bool = False) -> Tuple[pd.DataFrame, FELog]:
    """Generate polynomial / interaction features (synthetic creation)."""
    df = df.copy()
    use_cols = columns[:POLY_MAX_FEATURES]
    degree = min(degree, POLY_MAX_DEGREE)

    numeric_data = df[use_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    X_poly = poly.fit_transform(numeric_data.values)
    poly_names = poly.get_feature_names_out(use_cols)

    new_cols = []
    for i, name in enumerate(poly_names):
        clean = name.replace(" ", "_")
        if clean not in df.columns and clean not in use_cols:
            df[clean] = X_poly[:, i]
            new_cols.append(clean)

    return df, FELog(
        operation=f"Polynomial Features (degree={degree})",
        category="creation",
        columns_in=use_cols, columns_out=new_cols,
        detail=f"Generated {len(new_cols)} polynomial/interaction features from {len(use_cols)} columns.",
        rows_before=len(df), rows_after=len(df),
        cols_before=len(df.columns) - len(new_cols), cols_after=len(df.columns),
    )


def create_arithmetic(df: pd.DataFrame, col_a: str, col_b: str,
                       operations: List[str] = None) -> Tuple[pd.DataFrame, FELog]:
    """Arithmetic feature combinations: add, subtract, multiply, divide, ratio."""
    df = df.copy()
    if operations is None:
        operations = ["add", "subtract", "multiply", "divide"]

    a = pd.to_numeric(df[col_a], errors="coerce").fillna(0)
    b = pd.to_numeric(df[col_b], errors="coerce").fillna(0)
    new_cols = []
    op_map = {
        "add": (f"{col_a}_plus_{col_b}", a + b),
        "subtract": (f"{col_a}_minus_{col_b}", a - b),
        "multiply": (f"{col_a}_times_{col_b}", a * b),
        "divide": (f"{col_a}_div_{col_b}", np.where(b != 0, a / b, 0)),
    }
    for op in operations:
        if op in op_map:
            name, vals = op_map[op]
            df[name] = vals
            new_cols.append(name)

    return df, FELog(
        operation="Arithmetic Features", category="creation",
        columns_in=[col_a, col_b], columns_out=new_cols,
        detail=f"Created {len(new_cols)} arithmetic features: {', '.join(operations)}.",
        rows_before=len(df), rows_after=len(df),
        cols_before=len(df.columns) - len(new_cols), cols_after=len(df.columns),
    )


def create_aggregation(df: pd.DataFrame, columns: List[str],
                        agg_funcs: List[str] = None) -> Tuple[pd.DataFrame, FELog]:
    """Row-wise aggregation: mean, sum, std, min, max across selected columns."""
    df = df.copy()
    if agg_funcs is None:
        agg_funcs = ["mean", "sum", "std"]

    numeric = df[columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    prefix = "_".join(columns[:3]) + ("_etc" if len(columns) > 3 else "")
    new_cols = []

    func_map = {
        "mean": numeric.mean(axis=1),
        "sum": numeric.sum(axis=1),
        "std": numeric.std(axis=1).fillna(0),
        "min": numeric.min(axis=1),
        "max": numeric.max(axis=1),
        "range": numeric.max(axis=1) - numeric.min(axis=1),
    }
    for f in agg_funcs:
        if f in func_map:
            name = f"agg_{f}_{prefix}"
            df[name] = func_map[f]
            new_cols.append(name)

    return df, FELog(
        operation=f"Row Aggregation ({', '.join(agg_funcs)})", category="creation",
        columns_in=columns, columns_out=new_cols,
        detail=f"Created {len(new_cols)} aggregation features across {len(columns)} columns.",
        rows_before=len(df), rows_after=len(df),
        cols_before=len(df.columns) - len(new_cols), cols_after=len(df.columns),
    )


# ═══════════════════════════════════════════════════════
# 2. FEATURE TRANSFORMATION
# ═══════════════════════════════════════════════════════

def encode_label(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, FELog]:
    df = df.copy()
    new_cols = []
    mappings = {}
    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_label"] = le.fit_transform(df[col].astype(str))
            new_cols.append(f"{col}_label")
            mappings[col] = dict(zip(le.classes_.tolist(), range(len(le.classes_))))

    return df, FELog(
        operation="Label Encoding", category="transformation",
        columns_in=columns, columns_out=new_cols,
        detail=f"Label encoded {len(columns)} columns to integer labels.",
        rows_before=len(df), rows_after=len(df),
        cols_before=len(df.columns) - len(new_cols), cols_after=len(df.columns),
        intermediates={"mappings": mappings},
    )


def encode_onehot(df: pd.DataFrame, columns: List[str], drop_first: bool = True,
                   max_categories: int = 20) -> Tuple[pd.DataFrame, FELog]:
    df = df.copy()
    cols_before = len(df.columns)
    new_cols = []
    capped = {}

    for col in columns:
        if col not in df.columns:
            continue
        nunique = df[col].nunique()
        if nunique > max_categories:
            top = df[col].value_counts().head(max_categories).index
            df[col] = df[col].where(df[col].isin(top), other="__OTHER__")
            capped[col] = nunique

        dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first, dtype=int)
        new_cols.extend(dummies.columns.tolist())
        df = pd.concat([df, dummies], axis=1)

    return df, FELog(
        operation="One-Hot Encoding", category="transformation",
        columns_in=columns, columns_out=new_cols,
        detail=f"Created {len(new_cols)} dummy columns (drop_first={drop_first})."
               + (f" Capped: {capped}" if capped else ""),
        rows_before=len(df), rows_after=len(df),
        cols_before=cols_before, cols_after=len(df.columns),
    )


def encode_target(df: pd.DataFrame, columns: List[str], target_col: str) -> Tuple[pd.DataFrame, FELog]:
    df = df.copy()
    new_cols = []
    for col in columns:
        if col not in df.columns or target_col not in df.columns:
            continue
        means = df.groupby(col)[target_col].apply(lambda x: pd.to_numeric(x, errors="coerce").mean())
        name = f"{col}_target_enc"
        df[name] = df[col].map(means).astype(float).fillna(df[target_col].mean() if target_col in df else 0)
        new_cols.append(name)

    return df, FELog(
        operation="Target Encoding", category="transformation",
        columns_in=columns, columns_out=new_cols,
        detail=f"Target-encoded {len(columns)} columns using mean of '{target_col}'.",
        rows_before=len(df), rows_after=len(df),
        cols_before=len(df.columns) - len(new_cols), cols_after=len(df.columns),
    )


def encode_frequency(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, FELog]:
    df = df.copy()
    new_cols = []
    for col in columns:
        if col not in df.columns:
            continue
        freq = df[col].value_counts(normalize=True)
        name = f"{col}_freq"
        df[name] = df[col].map(freq).astype(float).fillna(0)
        new_cols.append(name)

    return df, FELog(
        operation="Frequency Encoding", category="transformation",
        columns_in=columns, columns_out=new_cols,
        detail=f"Frequency-encoded {len(columns)} columns.",
        rows_before=len(df), rows_after=len(df),
        cols_before=len(df.columns) - len(new_cols), cols_after=len(df.columns),
    )


def transform_log(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, FELog]:
    df = df.copy()
    new_cols = []
    for col in columns:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            name = f"{col}_log"
            df[name] = np.log1p(vals.clip(lower=0)).fillna(0)
            new_cols.append(name)

    return df, FELog(
        operation="Log Transform (log1p)", category="transformation",
        columns_in=columns, columns_out=new_cols,
        detail=f"Applied log1p to {len(columns)} skewed columns.",
        rows_before=len(df), rows_after=len(df),
        cols_before=len(df.columns) - len(new_cols), cols_after=len(df.columns),
    )


def transform_sqrt(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, FELog]:
    df = df.copy()
    new_cols = []
    for col in columns:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            name = f"{col}_sqrt"
            df[name] = np.sqrt(vals.clip(lower=0)).fillna(0)
            new_cols.append(name)

    return df, FELog(
        operation="Sqrt Transform", category="transformation",
        columns_in=columns, columns_out=new_cols,
        detail=f"Applied sqrt to {len(columns)} columns.",
        rows_before=len(df), rows_after=len(df),
        cols_before=len(df.columns) - len(new_cols), cols_after=len(df.columns),
    )


def transform_boxcox(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, FELog]:
    """Yeo-Johnson power transform (handles negative values unlike Box-Cox)."""
    df = df.copy()
    new_cols = []
    lambdas = {}
    for col in columns:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0).values.reshape(-1, 1)
        try:
            pt = PowerTransformer(method="yeo-johnson", standardize=False)
            transformed = pt.fit_transform(vals).ravel()
            name = f"{col}_boxcox"
            df[name] = transformed
            new_cols.append(name)
            lambdas[col] = round(float(pt.lambdas_[0]), 4)
        except Exception:
            continue

    return df, FELog(
        operation="Box-Cox (Yeo-Johnson) Transform", category="transformation",
        columns_in=columns, columns_out=new_cols,
        detail=f"Power-transformed {len(new_cols)} columns. Lambdas: {lambdas}",
        rows_before=len(df), rows_after=len(df),
        cols_before=len(df.columns) - len(new_cols), cols_after=len(df.columns),
        intermediates={"lambdas": lambdas},
    )


def transform_binning(df: pd.DataFrame, columns: List[str],
                       n_bins: int = 5, strategy: str = "quantile") -> Tuple[pd.DataFrame, FELog]:
    df = df.copy()
    new_cols = []
    n_bins = min(n_bins, MAX_BINS_DISCRETIZE)
    for col in columns:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        name = f"{col}_bin{n_bins}"
        try:
            if strategy == "quantile":
                df[name] = pd.qcut(vals, q=n_bins, labels=False, duplicates="drop")
            else:
                df[name] = pd.cut(vals, bins=n_bins, labels=False)
            df[name] = df[name].fillna(-1).astype(int)
            new_cols.append(name)
        except Exception:
            continue

    return df, FELog(
        operation=f"Binning ({strategy}, {n_bins} bins)", category="transformation",
        columns_in=columns, columns_out=new_cols,
        detail=f"Discretized {len(new_cols)} columns into {n_bins} bins.",
        rows_before=len(df), rows_after=len(df),
        cols_before=len(df.columns) - len(new_cols), cols_after=len(df.columns),
    )


# ═══════════════════════════════════════════════════════
# 3. FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════

def extract_pca(df: pd.DataFrame, columns: List[str],
                 n_components: int = 5) -> Tuple[pd.DataFrame, Dict, FELog]:
    """
    PCA dimensionality reduction.
    Returns: (df, pca_info, log)
    pca_info contains: explained_variance_ratio, cumulative_variance, component_loadings
    """
    df = df.copy()
    numeric = df[columns].apply(pd.to_numeric, errors="coerce").fillna(0)

    n_components = min(n_components, numeric.shape[1], numeric.shape[0])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric.values)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    new_cols = []
    for i in range(n_components):
        name = f"PC{i+1}"
        df[name] = X_pca[:, i]
        new_cols.append(name)

    # Loadings: which original features contribute most to each PC
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=columns
    )

    pca_info = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "n_components": n_components,
        "total_variance_explained": float(np.sum(pca.explained_variance_ratio_)),
        "loadings": loadings.to_dict(),
        "top_features_per_pc": {},
    }

    # Top contributing features per PC
    for i in range(n_components):
        abs_loadings = loadings[f"PC{i+1}"].abs().sort_values(ascending=False)
        pca_info["top_features_per_pc"][f"PC{i+1}"] = abs_loadings.head(5).to_dict()

    return df, pca_info, FELog(
        operation=f"PCA (n_components={n_components})", category="extraction",
        columns_in=columns, columns_out=new_cols,
        detail=f"Reduced {len(columns)} features to {n_components} PCs. "
               f"Total variance explained: {pca_info['total_variance_explained']:.1%}",
        rows_before=len(df), rows_after=len(df),
        cols_before=len(df.columns) - len(new_cols), cols_after=len(df.columns),
        intermediates=pca_info,
    )


def extract_datetime(df: pd.DataFrame, columns: List[str],
                      cyclical: bool = True) -> Tuple[pd.DataFrame, FELog]:
    df = df.copy()
    new_cols = []
    for col in columns:
        if col not in df.columns:
            continue
        try:
            dt = pd.to_datetime(df[col], errors="coerce")
            if dt.notna().sum() < len(df) * 0.3:
                continue
            for attr, name in [("year", f"{col}_year"), ("month", f"{col}_month"),
                                ("day", f"{col}_day"), ("dayofweek", f"{col}_dow"),
                                ("hour", f"{col}_hour")]:
                df[name] = getattr(dt.dt, attr).astype(float)
                new_cols.append(name)

            if cyclical:
                for attr, period, suffix in [("month", 12, "month"), ("dayofweek", 7, "dow"), ("hour", 24, "hour")]:
                    vals = getattr(dt.dt, attr).astype(float)
                    df[f"{col}_{suffix}_sin"] = np.sin(2 * np.pi * vals / period)
                    df[f"{col}_{suffix}_cos"] = np.cos(2 * np.pi * vals / period)
                    new_cols.extend([f"{col}_{suffix}_sin", f"{col}_{suffix}_cos"])

            for nc in new_cols:
                if nc in df.columns:
                    df[nc] = df[nc].fillna(0)
        except Exception:
            continue

    return df, FELog(
        operation="DateTime Extraction", category="extraction",
        columns_in=columns, columns_out=new_cols,
        detail=f"Extracted {len(new_cols)} datetime features (cyclical={cyclical}).",
        rows_before=len(df), rows_after=len(df),
        cols_before=len(df.columns) - len(new_cols), cols_after=len(df.columns),
    )


# ═══════════════════════════════════════════════════════
# 4. FEATURE SELECTION
# ═══════════════════════════════════════════════════════

# --- 4a. FILTER METHODS ---

def select_variance(df: pd.DataFrame, columns: List[str],
                     threshold: float = VARIANCE_THRESHOLD_DEFAULT) -> Tuple[pd.DataFrame, Dict, FELog]:
    df = df.copy()
    numeric = df[columns].select_dtypes(include=[np.number]).fillna(0)
    if numeric.empty:
        return df, {}, FELog(operation="Variance Threshold", category="selection",
            columns_in=columns, columns_out=[], detail="No numeric columns.",
            rows_before=len(df), rows_after=len(df),
            cols_before=len(df.columns), cols_after=len(df.columns))

    variances = numeric.var().sort_values()
    dropped = variances[variances < threshold].index.tolist()
    kept = variances[variances >= threshold].index.tolist()
    df = df.drop(columns=dropped)

    info = {"variances": variances.to_dict(), "dropped": dropped, "kept": kept, "threshold": threshold}

    return df, info, FELog(
        operation=f"Variance Threshold (≥{threshold})", category="selection",
        columns_in=columns, columns_out=dropped,
        detail=f"Dropped {len(dropped)} low-variance features. Kept {len(kept)}.",
        rows_before=len(df), rows_after=len(df),
        cols_before=len(df.columns) + len(dropped), cols_after=len(df.columns),
        intermediates=info,
    )


def select_correlation(df: pd.DataFrame, columns: List[str],
                         threshold: float = CORRELATION_DROP_THRESHOLD) -> Tuple[pd.DataFrame, Dict, FELog]:
    """Drop one of each pair of highly correlated features. Returns correlation matrix for viz."""
    df = df.copy()
    numeric = df[columns].select_dtypes(include=[np.number]).fillna(0)
    if numeric.shape[1] < 2:
        return df, {}, FELog(operation="Correlation Drop", category="selection",
            columns_in=columns, columns_out=[], detail="Need ≥2 numeric columns.",
            rows_before=len(df), rows_after=len(df),
            cols_before=len(df.columns), cols_after=len(df.columns))

    corr_matrix = numeric.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = []
    drop_reasons = {}
    for col in upper.columns:
        high_corr = upper[col][upper[col].abs() > threshold]
        if not high_corr.empty:
            to_drop.append(col)
            drop_reasons[col] = {idx: round(val, 4) for idx, val in high_corr.items()}

    df = df.drop(columns=to_drop, errors="ignore")

    info = {
        "correlation_matrix": corr_matrix.round(4).to_dict(),
        "dropped": to_drop,
        "drop_reasons": drop_reasons,
        "threshold": threshold,
        "n_features_before": numeric.shape[1],
        "n_features_after": numeric.shape[1] - len(to_drop),
    }

    return df, info, FELog(
        operation=f"Correlation Filter (>{threshold})", category="selection",
        columns_in=columns, columns_out=to_drop,
        detail=f"Dropped {len(to_drop)} correlated features. "
               + (f"Reasons: {'; '.join(f'{k} corr with {list(v.keys())[0]}' for k, v in list(drop_reasons.items())[:3])}" if drop_reasons else ""),
        rows_before=len(df), rows_after=len(df),
        cols_before=len(df.columns) + len(to_drop), cols_after=len(df.columns),
        intermediates=info,
    )


def select_mutual_info(df: pd.DataFrame, feature_cols: List[str],
                         target_col: str, task_type: str,
                         k: int = 10) -> Tuple[pd.DataFrame, Dict, FELog]:
    """Mutual information scoring — returns scores for all features + keeps top K."""
    df_work = df.copy()
    numeric = df_work[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    if numeric.empty or target_col not in df_work.columns:
        return df_work, {}, FELog(operation="Mutual Info", category="selection",
            columns_in=feature_cols, columns_out=[], detail="No valid features.",
            rows_before=len(df), rows_after=len(df),
            cols_before=len(df.columns), cols_after=len(df.columns))

    y = pd.to_numeric(df_work[target_col], errors="coerce").fillna(0).values
    func = mutual_info_classif if "classification" in task_type else mutual_info_regression
    mi = func(numeric.values, y.astype(int) if "classification" in task_type else y,
              random_state=RANDOM_STATE)

    scores = dict(zip(numeric.columns.tolist(), mi.tolist()))
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    top_k = [name for name, _ in sorted_scores[:k]]
    dropped = [name for name, _ in sorted_scores[k:]]
    non_num = [c for c in df_work.columns if c not in numeric.columns]
    df_out = df_work[non_num + top_k]

    info = {"scores": dict(sorted_scores), "top_k": top_k, "dropped": dropped, "k": k}

    return df_out, info, FELog(
        operation=f"Mutual Info Top-{k}", category="selection",
        columns_in=feature_cols, columns_out=top_k,
        detail=f"Selected top {k} features by MI. Top 3: {', '.join(f'{n}({s:.4f})' for n, s in sorted_scores[:3])}",
        rows_before=len(df), rows_after=len(df_out),
        cols_before=len(df.columns), cols_after=len(df_out.columns),
        intermediates=info,
    )


def select_statistical(df: pd.DataFrame, feature_cols: List[str],
                         target_col: str, task_type: str,
                         k: int = 10) -> Tuple[pd.DataFrame, Dict, FELog]:
    """Statistical test selection: ANOVA F-test (classification) or F-regression."""
    df_work = df.copy()
    numeric = df_work[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    if numeric.empty:
        return df_work, {}, FELog(operation="Statistical Test", category="selection",
            columns_in=feature_cols, columns_out=[], detail="No valid features.",
            rows_before=len(df), rows_after=len(df),
            cols_before=len(df.columns), cols_after=len(df.columns))

    y = pd.to_numeric(df_work[target_col], errors="coerce").fillna(0).values
    X = numeric.values
    k_actual = min(k, X.shape[1])

    if "classification" in task_type:
        score_func = f_classif
        test_name = "ANOVA F-test"
    else:
        score_func = f_regression
        test_name = "F-regression"

    selector = SelectKBest(score_func, k=k_actual)
    try:
        selector.fit(X, y.astype(int) if "classification" in task_type else y)
    except Exception:
        # Fallback: use MI
        return select_mutual_info(df, feature_cols, target_col, task_type, k)

    f_scores = selector.scores_
    p_values = selector.pvalues_ if selector.pvalues_ is not None else np.zeros_like(f_scores)
    mask = selector.get_support()

    selected = [c for c, m in zip(numeric.columns, mask) if m]
    dropped = [c for c, m in zip(numeric.columns, mask) if not m]

    scores_dict = {}
    for col, f, p in zip(numeric.columns, f_scores, p_values):
        scores_dict[col] = {"f_score": round(float(f), 4), "p_value": round(float(p), 6)}

    non_num = [c for c in df_work.columns if c not in numeric.columns]
    df_out = df_work[non_num + selected]

    info = {"test": test_name, "scores": scores_dict, "selected": selected, "dropped": dropped}

    return df_out, info, FELog(
        operation=f"{test_name} Top-{k_actual}", category="selection",
        columns_in=feature_cols, columns_out=selected,
        detail=f"Selected {len(selected)} features by {test_name}. Dropped {len(dropped)}.",
        rows_before=len(df), rows_after=len(df_out),
        cols_before=len(df.columns), cols_after=len(df_out.columns),
        intermediates=info,
    )


# --- 4b. WRAPPER METHOD (RFE) ---

def select_rfe(df: pd.DataFrame, feature_cols: List[str],
                target_col: str, task_type: str,
                n_features: int = 10) -> Tuple[pd.DataFrame, Dict, FELog]:
    """Recursive Feature Elimination — wrapper method."""
    df_work = df.copy()
    numeric = df_work[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    if numeric.empty:
        return df_work, {}, FELog(operation="RFE", category="selection",
            columns_in=feature_cols, columns_out=[], detail="No valid features.",
            rows_before=len(df), rows_after=len(df),
            cols_before=len(df.columns), cols_after=len(df.columns))

    y = pd.to_numeric(df_work[target_col], errors="coerce").fillna(0).values
    X = numeric.values
    n_features = min(n_features, X.shape[1])

    if "classification" in task_type:
        estimator = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
        y_fit = y.astype(int)
    else:
        estimator = RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
        y_fit = y.astype(float)

    rfe = RFE(estimator, n_features_to_select=n_features, step=1)
    rfe.fit(X, y_fit)

    rankings = dict(zip(numeric.columns.tolist(), rfe.ranking_.tolist()))
    mask = rfe.support_
    selected = [c for c, m in zip(numeric.columns, mask) if m]
    dropped = [c for c, m in zip(numeric.columns, mask) if not m]

    non_num = [c for c in df_work.columns if c not in numeric.columns]
    df_out = df_work[non_num + selected]

    info = {"rankings": rankings, "selected": selected, "dropped": dropped, "n_features": n_features}

    return df_out, info, FELog(
        operation=f"RFE (n_features={n_features})", category="selection",
        columns_in=feature_cols, columns_out=selected,
        detail=f"RFE selected {len(selected)} features using RandomForest. "
               f"Top: {', '.join(selected[:5])}",
        rows_before=len(df), rows_after=len(df_out),
        cols_before=len(df.columns), cols_after=len(df_out.columns),
        intermediates=info,
    )


# --- 4c. EMBEDDED METHODS ---

def select_lasso(df: pd.DataFrame, feature_cols: List[str],
                   target_col: str, task_type: str,
                   alpha: float = 0.01) -> Tuple[pd.DataFrame, Dict, FELog]:
    """L1 (Lasso) embedded feature selection — zero coefficients = irrelevant."""
    df_work = df.copy()
    numeric = df_work[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    if numeric.empty:
        return df_work, {}, FELog(operation="Lasso", category="selection",
            columns_in=feature_cols, columns_out=[], detail="No valid features.",
            rows_before=len(df), rows_after=len(df),
            cols_before=len(df.columns), cols_after=len(df.columns))

    scaler = StandardScaler()
    X = scaler.fit_transform(numeric.values)
    y = pd.to_numeric(df_work[target_col], errors="coerce").fillna(0).values

    if "classification" in task_type:
        model = LogisticRegression(penalty="l1", C=1.0/max(alpha, 1e-6), solver="saga",
                                    max_iter=1000, random_state=RANDOM_STATE)
        model.fit(X, y.astype(int))
        coefs = np.mean(np.abs(model.coef_), axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_.ravel())
    else:
        model = Lasso(alpha=alpha, random_state=RANDOM_STATE, max_iter=1000)
        model.fit(X, y)
        coefs = np.abs(model.coef_)

    coef_dict = dict(zip(numeric.columns.tolist(), coefs.tolist()))
    selected = [c for c, v in coef_dict.items() if v > 1e-6]
    dropped = [c for c, v in coef_dict.items() if v <= 1e-6]

    non_num = [c for c in df_work.columns if c not in numeric.columns]
    df_out = df_work[non_num + selected] if selected else df_work

    info = {"coefficients": coef_dict, "selected": selected, "dropped": dropped, "alpha": alpha}

    return df_out, info, FELog(
        operation=f"Lasso/L1 Selection (α={alpha})", category="selection",
        columns_in=feature_cols, columns_out=selected,
        detail=f"Lasso kept {len(selected)} features (non-zero coefficients). Dropped {len(dropped)}.",
        rows_before=len(df), rows_after=len(df_out),
        cols_before=len(df.columns), cols_after=len(df_out.columns),
        intermediates=info,
    )


def select_tree_importance(df: pd.DataFrame, feature_cols: List[str],
                            target_col: str, task_type: str,
                            threshold: float = 0.01) -> Tuple[pd.DataFrame, Dict, FELog]:
    """Tree-based embedded selection — features with importance > threshold."""
    df_work = df.copy()
    numeric = df_work[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    if numeric.empty:
        return df_work, {}, FELog(operation="Tree Importance", category="selection",
            columns_in=feature_cols, columns_out=[], detail="No valid features.",
            rows_before=len(df), rows_after=len(df),
            cols_before=len(df.columns), cols_after=len(df.columns))

    y = pd.to_numeric(df_work[target_col], errors="coerce").fillna(0).values

    if "classification" in task_type:
        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(numeric.values, y.astype(int))
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(numeric.values, y)

    importances = dict(zip(numeric.columns.tolist(), model.feature_importances_.tolist()))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    selected = [c for c, v in sorted_imp if v >= threshold]
    dropped = [c for c, v in sorted_imp if v < threshold]

    if not selected:
        selected = [sorted_imp[0][0]]  # Keep at least the most important
        dropped = [c for c, _ in sorted_imp[1:]]

    non_num = [c for c in df_work.columns if c not in numeric.columns]
    df_out = df_work[non_num + selected]

    info = {"importances": dict(sorted_imp), "selected": selected, "dropped": dropped, "threshold": threshold}

    return df_out, info, FELog(
        operation=f"Tree Importance (≥{threshold})", category="selection",
        columns_in=feature_cols, columns_out=selected,
        detail=f"RF importance kept {len(selected)} features. Top: {', '.join(f'{n}({v:.4f})' for n, v in sorted_imp[:3])}",
        rows_before=len(df), rows_after=len(df_out),
        cols_before=len(df.columns), cols_after=len(df_out.columns),
        intermediates=info,
    )


# --- 4d. AUTO SELECTOR (runs all methods, aggregates) ---

def auto_select_features(df: pd.DataFrame, feature_cols: List[str],
                          target_col: str, task_type: str,
                          top_k: int = 10) -> Tuple[pd.DataFrame, Dict, FELog]:
    """
    Runs ALL selection methods and aggregates rankings.
    Returns consensus top-K features with full intermediate results.
    """
    numeric = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    if numeric.empty:
        return df, {}, FELog(operation="Auto Select", category="selection",
            columns_in=feature_cols, columns_out=[], detail="No numeric features.",
            rows_before=len(df), rows_after=len(df),
            cols_before=len(df.columns), cols_after=len(df.columns))

    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).values
    X = numeric.values
    cols = numeric.columns.tolist()
    n_feat = len(cols)
    top_k = min(top_k, n_feat)

    all_rankings = {}  # method → {feature: rank}

    # 1. Mutual Information
    try:
        func = mutual_info_classif if "classification" in task_type else mutual_info_regression
        mi = func(X, y.astype(int) if "classification" in task_type else y, random_state=RANDOM_STATE)
        mi_sorted = sorted(zip(cols, mi), key=lambda x: x[1], reverse=True)
        all_rankings["Mutual Information"] = {c: rank + 1 for rank, (c, _) in enumerate(mi_sorted)}
    except Exception:
        pass

    # 2. Statistical Test
    try:
        score_func = f_classif if "classification" in task_type else f_regression
        f_scores, p_vals = score_func(X, y.astype(int) if "classification" in task_type else y)
        f_sorted = sorted(zip(cols, f_scores), key=lambda x: x[1], reverse=True)
        all_rankings["Statistical (ANOVA/F)"] = {c: rank + 1 for rank, (c, _) in enumerate(f_sorted)}
    except Exception:
        pass

    # 3. Correlation with target
    try:
        corrs = {}
        for i, c in enumerate(cols):
            corrs[c] = abs(float(np.corrcoef(X[:, i], y)[0, 1]))
        c_sorted = sorted(corrs.items(), key=lambda x: x[1], reverse=True)
        all_rankings["Correlation with Target"] = {c: rank + 1 for rank, (c, _) in enumerate(c_sorted)}
    except Exception:
        pass

    # 4. Tree Importance
    try:
        if "classification" in task_type:
            rf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
            rf.fit(X, y.astype(int))
        else:
            rf = RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
            rf.fit(X, y)
        imp_sorted = sorted(zip(cols, rf.feature_importances_), key=lambda x: x[1], reverse=True)
        all_rankings["Tree Importance"] = {c: rank + 1 for rank, (c, _) in enumerate(imp_sorted)}
    except Exception:
        pass

    # 5. Lasso
    try:
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        if "classification" in task_type:
            lm = LogisticRegression(penalty="l1", C=1.0, solver="saga", max_iter=500, random_state=RANDOM_STATE)
            lm.fit(X_s, y.astype(int))
            coefs = np.mean(np.abs(lm.coef_), axis=0) if lm.coef_.ndim > 1 else np.abs(lm.coef_.ravel())
        else:
            lm = Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=500)
            lm.fit(X_s, y)
            coefs = np.abs(lm.coef_)
        l_sorted = sorted(zip(cols, coefs), key=lambda x: x[1], reverse=True)
        all_rankings["Lasso (L1)"] = {c: rank + 1 for rank, (c, _) in enumerate(l_sorted)}
    except Exception:
        pass

    # Aggregate: average rank across all methods
    avg_ranks = {}
    for c in cols:
        ranks = [method_ranks.get(c, n_feat) for method_ranks in all_rankings.values()]
        avg_ranks[c] = np.mean(ranks) if ranks else n_feat

    final_sorted = sorted(avg_ranks.items(), key=lambda x: x[1])
    selected = [c for c, _ in final_sorted[:top_k]]
    dropped = [c for c, _ in final_sorted[top_k:]]

    non_num = [c for c in df.columns if c not in numeric.columns]
    df_out = df[non_num + selected].copy()

    info = {
        "method_rankings": all_rankings,
        "average_ranks": dict(final_sorted),
        "selected": selected,
        "dropped": dropped,
        "top_k": top_k,
        "n_methods_used": len(all_rankings),
    }

    return df_out, info, FELog(
        operation=f"Auto Feature Selection (Top-{top_k})", category="selection",
        columns_in=feature_cols, columns_out=selected,
        detail=f"Aggregated {len(all_rankings)} methods. Selected {len(selected)} features. "
               f"Best: {', '.join(selected[:3])}",
        rows_before=len(df), rows_after=len(df_out),
        cols_before=len(df.columns), cols_after=len(df_out.columns),
        intermediates=info,
    )


# ═══════════════════════════════════════════════════════
# 5. FEATURE SCALING
# ═══════════════════════════════════════════════════════

def _apply_scaler(df, columns, ScalerClass, name_suffix, op_name):
    df = df.copy()
    scaled_cols = []
    for col in columns:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
            new_name = f"{col}_{name_suffix}"
            scaler = ScalerClass()
            df[new_name] = scaler.fit_transform(vals.values.reshape(-1, 1)).ravel()
            scaled_cols.append(new_name)

    return df, FELog(
        operation=op_name, category="scaling",
        columns_in=columns, columns_out=scaled_cols,
        detail=f"Applied {op_name} to {len(scaled_cols)} columns.",
        rows_before=len(df), rows_after=len(df),
        cols_before=len(df.columns) - len(scaled_cols), cols_after=len(df.columns),
    )


def scale_standard(df, columns):
    return _apply_scaler(df, columns, StandardScaler, "scaled", "Standard Scaling (z-score)")

def scale_minmax(df, columns):
    return _apply_scaler(df, columns, MinMaxScaler, "minmax", "Min-Max Scaling [0,1]")

def scale_robust(df, columns):
    return _apply_scaler(df, columns, RobustScaler, "robust", "Robust Scaling (median/IQR)")

def scale_maxabs(df, columns):
    return _apply_scaler(df, columns, MaxAbsScaler, "maxabs", "MaxAbs Scaling [-1,1]")


# ═══════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════

def drop_columns(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, FELog]:
    df = df.copy()
    existing = [c for c in columns if c in df.columns]
    df = df.drop(columns=existing)
    return df, FELog(
        operation="Drop Columns", category="utility",
        columns_in=existing, columns_out=[],
        detail=f"Dropped {len(existing)} columns: {', '.join(existing[:8])}{'...' if len(existing) > 8 else ''}",
        rows_before=len(df), rows_after=len(df),
        cols_before=len(df.columns) + len(existing), cols_after=len(df.columns),
    )