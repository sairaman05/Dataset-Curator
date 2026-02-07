"""App-wide configuration and constants."""

APP_TITLE = "🧹 Dataset Curation Agent"
APP_ICON = "🧹"
APP_LAYOUT = "wide"

SUPPORTED_FILE_TYPES = ["csv", "xlsx", "xls", "json", "tsv", "parquet"]

# Cleaning thresholds
NULL_DROP_THRESHOLD = 0.7       # Drop column if >70% null
DUPLICATE_SUBSET = None         # None = check all columns
CATEGORICAL_THRESHOLD = 20      # If unique values < this, treat as categorical

# Display
MAX_PREVIEW_ROWS = 100
CHART_HEIGHT = 400