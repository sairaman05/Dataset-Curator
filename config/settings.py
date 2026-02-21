# config/settings.py
"""All application constants and thresholds."""

# ─── App ───────────────────────────────────────────────
APP_TITLE = "🧹 Dataset Curation Agent"
APP_ICON = "🧹"
APP_LAYOUT = "wide"
SUPPORTED_FILE_TYPES = ["csv", "xlsx", "xls", "json", "tsv", "parquet"]

# ─── Cleaning ──────────────────────────────────────────
NULL_DROP_THRESHOLD = 0.7
CATEGORICAL_THRESHOLD = 20

# ─── EDA ───────────────────────────────────────────────
OUTLIER_IQR_MULTIPLIER = 1.5
CORRELATION_STRONG_THRESHOLD = 0.7
CORRELATION_MODERATE_THRESHOLD = 0.4
SKEW_THRESHOLD = 1.0
IMBALANCE_RATIO_THRESHOLD = 0.6
HIGH_CARDINALITY_THRESHOLD = 50
LOW_VARIANCE_THRESHOLD = 0.01

# ─── ML ────────────────────────────────────────────────
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SIZE = 0.1
DEFAULT_EPOCHS = 50
MAX_EPOCHS = 1000
MIN_EPOCHS = 5
RANDOM_STATE = 42
CLASSIFICATION_UNIQUE_THRESHOLD = 20
CHART_HEIGHT = 400

# Models to SKIP from auto-discovery (unstable, deprecated, or meta-estimators)
SKIP_MODELS = {
    # Meta-estimators / wrappers (need sub-estimators)
    "AdaBoostClassifier", "AdaBoostRegressor",
    "BaggingClassifier", "BaggingRegressor",
    "StackingClassifier", "StackingRegressor",
    "VotingClassifier", "VotingRegressor",
    "MultiOutputClassifier", "MultiOutputRegressor",
    "ClassifierChain", "RegressorChain",
    "OneVsOneClassifier", "OneVsRestClassifier",
    "OutputCodeClassifier",
    # Require special input
    "CalibratedClassifierCV",
    "IsotonicRegression",
    "MultiTaskElasticNet", "MultiTaskElasticNetCV",
    "MultiTaskLasso", "MultiTaskLassoCV",
    "RadiusNeighborsClassifier", "RadiusNeighborsRegressor",
    # Dummy / baseline
    "DummyClassifier", "DummyRegressor",
    # Unstable or niche
    "GaussianProcessClassifier", "GaussianProcessRegressor",
    "PLSRegression", "PLSCanonical",
    "NuSVC", "NuSVR",
    "QuantileRegressor",
    # Deprecated
    "PassiveAggressiveClassifier", "PassiveAggressiveRegressor",
}

# Friendly display names and descriptions for common models
MODEL_DESCRIPTIONS = {
    "LogisticRegression": {
        "name": "Logistic Regression",
        "desc": "Linear model for classification using logistic function. Fast, interpretable, works well as a baseline.",
        "complexity": "Low",
        "arch": "Applies a linear transformation (w·x + b) followed by a sigmoid activation to produce class probabilities. Uses L2 regularization by default to prevent overfitting."
    },
    "RandomForestClassifier": {
        "name": "Random Forest Classifier",
        "desc": "Ensemble of decision trees with bagging. Robust, handles mixed features, good out-of-box performance.",
        "complexity": "Medium",
        "arch": "Constructs N decision trees, each trained on a bootstrap sample of the data with random feature subsets. Final prediction is majority vote across all trees."
    },
    "RandomForestRegressor": {
        "name": "Random Forest Regressor",
        "desc": "Ensemble of decision trees for regression. Robust to outliers and non-linear relationships.",
        "complexity": "Medium",
        "arch": "Constructs N decision trees on bootstrap samples. Final prediction is the mean of all tree predictions."
    },
    "GradientBoostingClassifier": {
        "name": "Gradient Boosting Classifier",
        "desc": "Sequential ensemble that corrects errors iteratively. Strong performance on structured data.",
        "complexity": "Medium-High",
        "arch": "Builds trees sequentially where each new tree corrects residual errors of the ensemble so far. Uses gradient descent on the loss function to determine corrections."
    },
    "GradientBoostingRegressor": {
        "name": "Gradient Boosting Regressor",
        "desc": "Sequential boosting for regression. Excellent for structured/tabular data.",
        "complexity": "Medium-High",
        "arch": "Sequentially adds decision trees, each fitting the negative gradient (residuals) of the loss function from the current ensemble."
    },
    "XGBClassifier": {
        "name": "XGBoost Classifier",
        "desc": "Optimized gradient boosting with regularization. Top performer in competitions.",
        "complexity": "High",
        "arch": "Advanced gradient boosting with L1/L2 regularization on leaf weights, column subsampling, and Newton-Raphson optimization using second-order gradients (Hessian)."
    },
    "XGBRegressor": {
        "name": "XGBoost Regressor",
        "desc": "Optimized gradient boosting for regression. Fast, regularized, highly tunable.",
        "complexity": "High",
        "arch": "Uses second-order Taylor expansion of the loss function for optimal split finding, with regularized leaf weights and shrinkage."
    },
    "LGBMClassifier": {
        "name": "LightGBM Classifier",
        "desc": "Gradient boosting with histogram-based splits. Very fast on large datasets.",
        "complexity": "High",
        "arch": "Uses Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) for efficiency. Grows trees leaf-wise rather than level-wise."
    },
    "LGBMRegressor": {
        "name": "LightGBM Regressor",
        "desc": "Fast histogram-based gradient boosting for regression.",
        "complexity": "High",
        "arch": "Leaf-wise tree growth with histogram binning for features. GOSS keeps instances with large gradients, drops small gradient instances."
    },
    "SVC": {
        "name": "Support Vector Classifier",
        "desc": "Finds optimal hyperplane separating classes. Effective in high-dimensional spaces.",
        "complexity": "Medium-High",
        "arch": "Maps data to higher-dimensional space via kernel function, finds maximum-margin hyperplane. Uses hinge loss with C regularization parameter."
    },
    "SVR": {
        "name": "Support Vector Regressor",
        "desc": "SVM for regression with epsilon-insensitive loss. Good for non-linear regression.",
        "complexity": "Medium-High",
        "arch": "Finds a tube of width epsilon around the regression function. Only penalizes predictions outside this tube (epsilon-insensitive loss)."
    },
    "KNeighborsClassifier": {
        "name": "K-Nearest Neighbors Classifier",
        "desc": "Instance-based learning, classifies by majority vote of k nearest neighbors.",
        "complexity": "Low",
        "arch": "No explicit training phase. At prediction, computes distance to all training points and takes majority vote of k closest neighbors."
    },
    "KNeighborsRegressor": {
        "name": "K-Nearest Neighbors Regressor",
        "desc": "Predicts by averaging k nearest neighbors. Simple, non-parametric.",
        "complexity": "Low",
        "arch": "At prediction, finds k nearest training examples by distance metric and returns their average target value."
    },
    "DecisionTreeClassifier": {
        "name": "Decision Tree Classifier",
        "desc": "Single tree that splits data recursively. Highly interpretable.",
        "complexity": "Low",
        "arch": "Recursively splits feature space using information gain (entropy) or Gini impurity. Each leaf node contains a class prediction."
    },
    "DecisionTreeRegressor": {
        "name": "Decision Tree Regressor",
        "desc": "Single decision tree for regression. Fast, interpretable, but prone to overfitting.",
        "complexity": "Low",
        "arch": "Recursively partitions feature space to minimize MSE. Leaf nodes predict the mean of training samples in that partition."
    },
    "LinearRegression": {
        "name": "Linear Regression",
        "desc": "Ordinary least squares regression. The simplest baseline for regression tasks.",
        "complexity": "Low",
        "arch": "Finds coefficients w that minimize the sum of squared residuals: min ||Xw - y||². Closed-form solution via normal equation or gradient descent."
    },
    "Ridge": {
        "name": "Ridge Regression",
        "desc": "Linear regression with L2 regularization. Prevents overfitting with many features.",
        "complexity": "Low",
        "arch": "Minimizes ||Xw - y||² + α||w||² where α controls regularization strength. Shrinks coefficients toward zero without eliminating them."
    },
    "Lasso": {
        "name": "Lasso Regression",
        "desc": "Linear regression with L1 regularization. Performs feature selection by zeroing coefficients.",
        "complexity": "Low",
        "arch": "Minimizes ||Xw - y||² + α||w||₁. L1 penalty drives some coefficients exactly to zero, performing implicit feature selection."
    },
    "ElasticNet": {
        "name": "Elastic Net",
        "desc": "Combines L1 and L2 regularization. Balances Lasso's sparsity with Ridge's stability.",
        "complexity": "Low",
        "arch": "Minimizes ||Xw - y||² + α·ρ||w||₁ + α·(1-ρ)/2·||w||². Ratio ρ controls L1 vs L2 balance."
    },
    "ExtraTreesClassifier": {
        "name": "Extra Trees Classifier",
        "desc": "Extremely randomized trees. Faster than Random Forest with comparable accuracy.",
        "complexity": "Medium",
        "arch": "Like Random Forest but uses random thresholds for splitting instead of best thresholds. This adds more randomness, reducing variance."
    },
    "ExtraTreesRegressor": {
        "name": "Extra Trees Regressor",
        "desc": "Extremely randomized trees for regression. Fast with good generalization.",
        "complexity": "Medium",
        "arch": "Ensemble of trees with random split thresholds. Predictions are averaged across all trees."
    },
    "HistGradientBoostingClassifier": {
        "name": "Histogram Gradient Boosting Classifier",
        "desc": "Sklearn's native histogram-based boosting. Fast, handles missing values natively.",
        "complexity": "High",
        "arch": "Bins continuous features into 256 histogram bins for fast split finding. Supports native missing value handling and early stopping."
    },
    "HistGradientBoostingRegressor": {
        "name": "Histogram Gradient Boosting Regressor",
        "desc": "Native histogram-based boosting for regression. Efficient on large datasets.",
        "complexity": "High",
        "arch": "Histogram-binned gradient boosting with 256 bins per feature. Leaf-wise growth with L2 regularization."
    },
    "SGDClassifier": {
        "name": "SGD Classifier",
        "desc": "Linear classifier with stochastic gradient descent. Scales to very large datasets.",
        "complexity": "Low",
        "arch": "Fits a linear model using SGD optimization. Supports various loss functions (hinge for SVM, log_loss for logistic regression)."
    },
    "SGDRegressor": {
        "name": "SGD Regressor",
        "desc": "Linear regressor with SGD optimization. Memory efficient for large data.",
        "complexity": "Low",
        "arch": "Fits linear model via SGD with configurable loss (squared_error, huber, epsilon_insensitive)."
    },
    "GaussianNB": {
        "name": "Gaussian Naive Bayes",
        "desc": "Probabilistic classifier assuming feature independence. Very fast, good baseline.",
        "complexity": "Low",
        "arch": "Applies Bayes theorem with Gaussian likelihood: P(y|X) ∝ P(X|y)·P(y), assuming features are independent given the class."
    },
    "BernoulliNB": {
        "name": "Bernoulli Naive Bayes",
        "desc": "Naive Bayes for binary features. Works well for text classification.",
        "complexity": "Low",
        "arch": "Bayes classifier designed for binary/boolean features with Bernoulli distribution likelihood."
    },
    "MLPClassifier": {
        "name": "MLP Classifier (Neural Network)",
        "desc": "Multi-layer perceptron. Can learn non-linear decision boundaries.",
        "complexity": "High",
        "arch": "Feedforward neural network with configurable hidden layers. Uses backpropagation with ReLU activation and Adam optimizer by default."
    },
    "MLPRegressor": {
        "name": "MLP Regressor (Neural Network)",
        "desc": "Neural network for regression. Handles complex non-linear patterns.",
        "complexity": "High",
        "arch": "Feedforward neural network with backpropagation. Output layer has identity activation for continuous predictions."
    },
}