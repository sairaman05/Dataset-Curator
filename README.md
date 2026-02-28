# 🧹 Dataset Curation Agent

An end-to-end automated data science pipeline built with Streamlit that transforms raw, messy datasets into actionable insights, engineered features, and trained ML models — all without writing a single line of code.

Upload any tabular dataset and walk through a complete data science workflow: **Auto-Clean → EDA & Insights → Feature Engineering → ML Training & Evaluation**.


## Features

- **One-Click Data Cleaning** — Automatic null handling, duplicate removal, dtype inference, datetime detection, and categorical encoding
- **Automated EDA** — Statistical profiling, outlier detection, correlation analysis, skewness analysis, class imbalance detection, and natural-language insight storytelling
- **Interactive Feature Engineering** — 28 operations across 5 categories (creation, transformation, extraction, selection, scaling) with undo/redo support
- **Auto Feature Selector** — Runs 5 selection methods simultaneously (Mutual Info, ANOVA, Correlation, Random Forest, Lasso) and aggregates consensus rankings with full visualizations
- **PCA Dimensionality Reduction** — With explained variance charts and per-component loading analysis
- **54 ML Models** — Auto-discovered from scikit-learn, XGBoost, and LightGBM (22 classifiers + 32 regressors)
- **Smart Model Recommendations** — Analyzes your data and recommends top 5 models with scoring rationale
- **Live Training Dashboard** — Real-time epoch-by-epoch progress with train/validation curves
- **Model Comparison** — Side-by-side training and evaluation of two models with winner declaration
- **Downloadable Reports** — Markdown training reports, trained model (.pkl) files, cleaned/engineered data (CSV/Excel)
- **Bulletproof Training** — Safe stratification fallback, automatic dtype fixes, friendly error messages for every edge case

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌───────────┐  │
│  │  Clean &  │  │  EDA &   │  │   Feature    │  │    ML     │  │
│  │  Profile  │  │ Insights │  │ Engineering  │  │ Training  │  │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘  └─────┬─────┘  │
│       │              │               │                │        │
│  ┌────▼─────┐  ┌────▼─────┐  ┌──────▼───────┐  ┌─────▼─────┐ │
│  │ summary  │  │ eda_view │  │   fe_view    │  │  ml_view  │ │
│  │ viz_view │  │          │  │              │  │           │ │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘  └─────┬─────┘ │
└───────┼──────────────┼───────────────┼────────────────┼────────┘
        │              │               │                │
┌───────▼──────────────▼───────────────▼────────────────▼────────┐
│                         Agents Layer                            │
│  ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │ Curation   │  │   EDA    │  │    FE    │  │     ML      │  │
│  │   Agent    │  │  Agent   │  │  Agent   │  │    Agent    │  │
│  └────┬───────┘  └────┬─────┘  └────┬─────┘  └──────┬──────┘  │
└───────┼──────────────┼──────────────┼───────────────┼──────────┘
        │              │              │               │
┌───────▼──────────────▼──────────────▼───────────────▼──────────┐
│                          Core Layer                             │
│  ┌──────────┐ ┌───────────┐ ┌──────────────┐ ┌─────────────┐  │
│  │  Loader  │ │  Insight  │ │   Feature    │ │  Registry   │  │
│  │ Cleaner  │ │  Engine   │ │  Engineer    │ │  Analyzer   │  │
│  │ Profiler │ │   Story   │ │  (930 lines) │ │  Trainer    │  │
│  │ Viz      │ │ Generator │ │              │ │  Evaluator  │  │
│  └──────────┘ └───────────┘ └──────────────┘ │  Comparator │  │
│                                               │  Reports    │  │
│                                               └─────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.9+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Dataset-Curation-Agent.git
cd Dataset-Curation-Agent

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Requirements

```
streamlit>=1.30.0
pandas>=2.1.0
numpy>=1.24.0
openpyxl>=3.1.0
plotly>=5.18.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
```

---

## Usage

1. **Launch the app**: `streamlit run app.py`
2. **Upload a dataset** (CSV, Excel, JSON, TSV, or Parquet) via the sidebar
3. **Click "Run Analysis"** — cleaning and EDA run automatically
4. **Navigate through the 4 tabs** to clean, explore, engineer features, and train models
5. **Download results** — cleaned data, engineered data, trained models, and evaluation reports

### Supported File Formats

| Format | Extensions |
|--------|-----------|
| CSV | `.csv` |
| Excel | `.xlsx`, `.xls` |
| JSON | `.json` |
| TSV | `.tsv` |
| Parquet | `.parquet` |

---

## Pipeline Walkthrough

### Tab 1: Clean & Profile

Automatic data cleaning with detailed profiling and visualizations.

**What it does:**
- Detects and handles null values (drops columns above threshold, fills remaining)
- Removes duplicate rows
- Infers and converts data types (numeric, categorical, datetime)
- Converts low-cardinality numeric columns to categorical
- Profiles every column: stats, distributions, value counts

**What you see:**
- Before/after overview metrics (rows, columns, nulls, memory)
- Cleaning log showing every action taken
- Null comparison chart (before vs after)
- Column info table with dtypes
- Numeric statistics (mean, std, quartiles, skewness, kurtosis)
- Categorical statistics (unique values, top categories, frequency)
- Auto-generated distribution and bar charts
- Export buttons (CSV / Excel)

---

### Tab 2: EDA & Insights

Automated exploratory data analysis with natural-language storytelling.

**What it detects:**
- Outliers (IQR method with configurable multiplier)
- Strong and moderate correlations between features
- Skewed distributions with transformation recommendations
- Class imbalance in categorical targets
- High-cardinality features
- Low-variance / near-constant features

**What you see:**
- Executive summary with severity-scored findings
- Each finding shown with context, impact, and recommendation
- Actionable recommendations grouped by priority
- Downloadable EDA report

---

### Tab 3: Feature Engineering

Interactive feature engineering with 28 operations, full visualization of intermediates, undo/redo, and one-click transfer to ML.

#### ① Feature Creation

| Method | Description |
|--------|-------------|
| Polynomial (Synthetic) | Generates `x²`, `x×y`, up to degree 3. Caps at 10 input features to prevent explosion |
| Arithmetic (Combination) | Creates `A+B`, `A−B`, `A×B`, `A÷B` between any two columns |
| Aggregation (Row-wise) | Computes mean, sum, std, min, max, range across selected columns per row |

#### ② Feature Transformation

**Encoding (Categorical → Numeric):**

| Method | Description |
|--------|-------------|
| One-Hot Encoding | Creates dummy variables with drop-first option and max category cap |
| Label Encoding | Maps categories to integers (shows mapping table) |
| Frequency Encoding | Replaces category with its occurrence proportion |
| Target Encoding | Replaces category with mean of target variable per category |

**Mathematical Transforms:**

| Method | Description |
|--------|-------------|
| Log (log1p) | Reduces right skew, handles zeros |
| Square Root | Milder than log for moderate skew |
| Box-Cox (Yeo-Johnson) | Finds optimal power transform, handles negatives, shows lambda values |
| Binning / Discretize | Converts numeric to bins (quantile or uniform strategy) |

#### ③ Feature Extraction

| Method | Visualizations |
|--------|---------------|
| PCA (Dimensionality Reduction) | Explained variance bar chart + cumulative line, per-component loading bar charts showing which features drive each PC, variance summary table |
| DateTime Extraction | Extracts year, month, day, day-of-week, hour + optional cyclical sin/cos encoding |

#### ④ Feature Selection

**🚀 Auto Feature Selector** — One-button runs 5 methods simultaneously and aggregates consensus rankings:

1. **Mutual Information** — Non-linear dependency measure
2. **ANOVA F-test / F-regression** — Statistical significance test
3. **Correlation with Target** — Linear relationship strength
4. **Random Forest Importance** — Gini impurity-based ranking
5. **Lasso (L1) Coefficient** — Regularization-based ranking

Produces a grouped bar chart comparing all method rankings with an average rank line, selected/dropped feature lists, and per-method detail tables.

**Individual Methods:**

| Category | Method | Intermediate Visualizations |
|----------|--------|---------------------------|
| Filter | Variance Threshold | Feature variance bar chart |
| Filter | Correlation Filter | Full correlation heatmap + drop reason explanations |
| Filter | Mutual Information | MI score bar chart |
| Filter | Statistical Test (ANOVA/F) | F-score bar chart + p-value table with significance flags |
| Wrapper | RFE (Recursive Feature Elimination) | Ranking bar chart + selected/eliminated lists |
| Embedded | Tree Importance (Random Forest) | Importance bar chart + kept/dropped with scores |
| Embedded | Lasso (L1 Regularization) | Coefficient magnitude bar chart + zero-coefficient list |

#### ⑤ Feature Scaling

| Method | Description |
|--------|-------------|
| Standard Scaling (z-score) | Mean=0, variance=1 |
| Min-Max Scaling | Rescales to [0, 1] |
| Robust Scaling | Uses median/IQR, resistant to outliers |
| MaxAbs Scaling | Scales to [-1, 1] by maximum absolute value |

**Additional features:**
- Drop Columns utility
- Full operation log with category badges (creation/transformation/extraction/selection/scaling)
- Undo last / Reset all buttons
- Data preview (first 50 rows)
- Download CSV / Excel
- **"Send to ML Training →"** button that transfers engineered data to Tab 4

---

### Tab 4: ML Training

Automated model discovery, smart recommendations, live training, and model comparison.

#### Target Analysis

Select your target column and click "Analyze" to get:
- Auto-detected task type (binary classification, multiclass classification, regression)
- Class distribution analysis and imbalance detection
- Feature count and data shape summary
- Top 5 model recommendations with scores and architecture descriptions

#### Single Model Training

- Choose from 54 auto-discovered models (22 classifiers, 32 regressors)
- Set epoch count (5–1000)
- Live training dashboard: progress bar, epoch counter, train/val score cards, elapsed time
- Real-time training curve chart (accuracy/R² over epochs)
- Post-training results: metrics table, training progress chart, confusion matrix (classification) or actual-vs-predicted scatter (regression), feature importance bar chart
- Download: training report (.md) + trained model (.pkl)

#### Model Comparison

- Select Model A and Model B
- Both train sequentially with independent live progress
- Side-by-side comparison: winner announcement, metrics comparison bar chart, epoch curve overlay, confusion matrices for both models, epoch history tables
- Download: comparison report + individual reports + both models (.pkl)

#### Supported Models

**Classifiers (22):**
Logistic Regression, Random Forest, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, SVM (SVC/LinearSVC), KNN, Decision Tree, Extra Trees, MLP Neural Network, Naive Bayes (Gaussian/Bernoulli), LDA, Ridge Classifier, SGD Classifier, Perceptron, Nearest Centroid, Label Propagation, Label Spreading

**Regressors (32):**
Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, SVR (SVR/LinearSVR), KNN, Decision Tree, Extra Trees, MLP Neural Network, Bayesian Ridge, ARD, Huber, RANSAC, TheilSen, Lars, LassoLars, Kernel Ridge, Tweedie, and more

---

## Project Structure

```
Dataset_Curation_Agent/
├── app.py                          # Main Streamlit application (4 tabs)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── config/
│   └── settings.py                 # All constants, thresholds, model skip list (257 lines)
│
├── agents/                         # Orchestrators
│   ├── curation_agent.py           # Sprint 1: cleaning pipeline orchestrator
│   ├── eda_agent.py                # Sprint 2: EDA pipeline orchestrator
│   ├── fe_agent.py                 # Sprint 3: feature engineering orchestrator (250 lines)
│   └── ml_agent.py                 # Sprint 4: ML training orchestrator (131 lines)
│
├── core/                           # Business logic
│   ├── data_loader.py              # File parsing (CSV, Excel, JSON, TSV, Parquet)
│   ├── data_cleaner.py             # Null handling, dedup, dtype conversion
│   ├── data_profiler.py            # Statistical profiling
│   ├── data_visualizer.py          # Auto-generated distribution charts
│   ├── insight_engine.py           # Outlier, correlation, skew, imbalance detection
│   ├── story_generator.py          # Natural-language insight narratives
│   ├── feature_engineer.py         # 28 FE operations across 5 categories (930 lines)
│   ├── model_registry.py           # Auto-discovers sklearn/XGB/LGBM models (199 lines)
│   ├── model_analyzer.py           # Data analysis + model recommendation (251 lines)
│   ├── model_trainer.py            # Epoch-based training with safe wrappers (559 lines)
│   ├── model_evaluator.py          # Metrics computation + chart generation (276 lines)
│   ├── model_comparator.py         # Side-by-side model comparison (102 lines)
│   └── report_generator.py         # Markdown report generation (305 lines)
│
├── ui/                             # Streamlit UI components
│   ├── styles.py                   # Custom CSS
│   ├── sidebar.py                  # File upload + settings sidebar
│   ├── summary_view.py             # Sprint 1: cleaning results display
│   ├── visualization_view.py       # Sprint 1: auto-generated charts
│   ├── eda_view.py                 # Sprint 2: EDA findings + recommendations
│   ├── fe_view.py                  # Sprint 3: feature engineering UI (725 lines)
│   └── ml_view.py                  # Sprint 4: ML training UI (578 lines)
│
├── utils/
│   └── helpers.py                  # Shared utilities (safe_to_float, nuke_datetime, etc.)
│
├── data/                           # Uploaded datasets (gitignored)
└── reports/                        # Generated reports (gitignored)
```

**Total codebase: ~5,000 lines of Python across 19 modules.**

---

## Technical Details

### Data Safety

- **Zero pandas in sklearn**: Every feature column is individually converted to `numpy float64` via `safe_to_float()` before being passed to any model. This prevents all dtype errors.
- **Safe stratification**: If stratified train/test split fails (classes with <2 members), automatically falls back to non-stratified split.
- **Safe model instantiation**: `_safe_instantiate()` patches known problematic models (HistGradientBoosting early_stopping, TransformedTargetRegressor, etc.).
- **Safe fit**: `_safe_fit()` wraps every `model.fit()` call with automatic dtype correction and friendly error messages.

### Training Strategies

The trainer dispatches each model to the optimal training strategy:

| Strategy | Models | How |
|----------|--------|-----|
| Ensemble (warm_start) | RF, ExtraTrees, GradientBoosting | Incrementally adds trees, reports after each checkpoint |
| XGBoost native | XGBClassifier/Regressor | Uses `eval_set` for native loss tracking |
| LightGBM native | LGBMClassifier/Regressor | Uses `eval_set` for native loss tracking |
| Iterative (warm_start) | MLP, SGD, HistGBT | Epoch-by-epoch training with warm_start |
| Partial fit | Naive Bayes, Perceptron | Incremental `partial_fit()` calls |
| Single shot (simulated) | KNN, SVM, Ridge, etc. | Trains once, simulates progress curve |

### Feature Engineering Pipeline

The FE agent maintains a snapshot stack for undo/redo:
```
Apply transform → save snapshot → update current_df → log operation
Undo → pop snapshot → restore current_df → remove last log
Reset → restore original_df → clear all logs and snapshots
```

All 28 operations return `(DataFrame, FELog)` or `(DataFrame, info_dict, FELog)` where `info_dict` contains intermediate data for visualizations.

---

## Configuration

All thresholds and constants are centralized in `config/settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `NULL_DROP_THRESHOLD` | 0.7 | Drop columns with >70% nulls |
| `CATEGORICAL_THRESHOLD` | 20 | Numeric columns with ≤20 unique values → categorical |
| `OUTLIER_IQR_MULTIPLIER` | 1.5 | IQR multiplier for outlier detection |
| `CORRELATION_STRONG_THRESHOLD` | 0.7 | Strong correlation flag |
| `IMBALANCE_RATIO_THRESHOLD` | 0.6 | Class imbalance detection |
| `DEFAULT_TEST_SIZE` | 0.2 | Test set proportion |
| `DEFAULT_VAL_SIZE` | 0.1 | Validation set proportion |
| `RANDOM_STATE` | 42 | Global random seed |
| `POLY_MAX_FEATURES` | 10 | Max features for polynomial generation |
| `CORRELATION_DROP_THRESHOLD` | 0.95 | Drop one of correlated pair |
| `VARIANCE_THRESHOLD_DEFAULT` | 0.01 | Drop near-constant features |

---

## Troubleshooting

### Common Issues

**"Module not found" errors after updating files:**
```powershell
# Windows: clear Python bytecode cache
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
```

**Streamlit port already in use:**
```bash
streamlit run app.py --server.port 8502
```

**XGBoost/LightGBM not installed:**
```bash
pip install xgboost lightgbm
```
The app still works without them — those models simply won't appear in the registry.

**Training crashes on specific models:**
Some models have strict requirements. The system automatically skips 15+ known problematic models and wraps all training in safe error handling. If a model fails, you'll see a friendly error message suggesting alternatives.


## License

This project is developed for academic and learning purposes.