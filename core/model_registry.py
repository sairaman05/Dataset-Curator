# core/model_registry.py
"""
Dynamic Model Registry — Auto-discovers ALL available models.

Uses sklearn.utils.all_estimators() to discover every classifier and regressor
installed in the environment. Also registers XGBoost and LightGBM manually.
No hardcoded model lists — everything is discovered at runtime.
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Type

from sklearn.utils import all_estimators
from config.settings import SKIP_MODELS, MODEL_DESCRIPTIONS


@dataclass
class RegisteredModel:
    """A model discovered by the registry."""
    class_name: str                    # e.g., "RandomForestClassifier"
    display_name: str                  # e.g., "Random Forest Classifier"
    description: str
    architecture: str
    complexity: str                    # Low / Medium / Medium-High / High
    task_type: str                     # "classifier" or "regressor"
    source: str                        # "sklearn", "xgboost", "lightgbm"
    estimator_class: Type              # The actual class
    supports_warm_start: bool = False
    supports_n_estimators: bool = False
    supports_max_iter: bool = False
    supports_partial_fit: bool = False
    default_params: Dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """
    Auto-discovers all available ML models from sklearn, XGBoost, and LightGBM.

    Usage:
        registry = ModelRegistry()
        classifiers = registry.get_classifiers()
        regressors = registry.get_regressors()
        model_info = registry.get_model("RandomForestClassifier")
    """

    def __init__(self):
        self._models: Dict[str, RegisteredModel] = {}
        self._discover_sklearn_models()
        self._register_xgboost()
        self._register_lightgbm()

    def _discover_sklearn_models(self):
        """Use sklearn.utils.all_estimators() to find all classifiers and regressors."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Discover classifiers
            for name, cls in all_estimators(type_filter="classifier"):
                if name in SKIP_MODELS:
                    continue
                self._register_model(name, cls, "classifier", "sklearn")

            # Discover regressors
            for name, cls in all_estimators(type_filter="regressor"):
                if name in SKIP_MODELS:
                    continue
                self._register_model(name, cls, "regressor", "sklearn")

    def _register_xgboost(self):
        """Register XGBoost models."""
        try:
            from xgboost import XGBClassifier, XGBRegressor
            self._register_model("XGBClassifier", XGBClassifier, "classifier", "xgboost",
                                 default_params={"n_estimators": 100, "use_label_encoder": False,
                                                  "eval_metric": "logloss", "verbosity": 0})
            self._register_model("XGBRegressor", XGBRegressor, "regressor", "xgboost",
                                 default_params={"n_estimators": 100, "verbosity": 0})
        except ImportError:
            pass

    def _register_lightgbm(self):
        """Register LightGBM models."""
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
            self._register_model("LGBMClassifier", LGBMClassifier, "classifier", "lightgbm",
                                 default_params={"n_estimators": 100, "verbose": -1})
            self._register_model("LGBMRegressor", LGBMRegressor, "regressor", "lightgbm",
                                 default_params={"n_estimators": 100, "verbose": -1})
        except ImportError:
            pass

    def _register_model(self, name: str, cls: Type, task_type: str, source: str,
                        default_params: Optional[Dict] = None):
        """Register a single model with metadata."""
        # Check if model can be instantiated
        try:
            test_instance = cls() if not default_params else cls(**default_params)
        except Exception:
            return

        # Get description from our curated descriptions, or generate a generic one
        meta = MODEL_DESCRIPTIONS.get(name, {})
        display_name = meta.get("name", self._make_display_name(name))
        description = meta.get("desc", f"{display_name} from {source}.")
        architecture = meta.get("arch", f"Standard {display_name} implementation from {source}.")
        complexity = meta.get("complexity", "Medium")

        # Detect capabilities
        supports_warm_start = hasattr(test_instance, "warm_start")
        supports_n_estimators = hasattr(test_instance, "n_estimators")
        supports_max_iter = hasattr(test_instance, "max_iter")
        supports_partial_fit = hasattr(test_instance, "partial_fit")

        self._models[name] = RegisteredModel(
            class_name=name,
            display_name=display_name,
            description=description,
            architecture=architecture,
            complexity=complexity,
            task_type=task_type,
            source=source,
            estimator_class=cls,
            supports_warm_start=supports_warm_start,
            supports_n_estimators=supports_n_estimators,
            supports_max_iter=supports_max_iter,
            supports_partial_fit=supports_partial_fit,
            default_params=default_params or {},
        )

    def _make_display_name(self, class_name: str) -> str:
        """Convert CamelCase to readable name."""
        import re
        name = re.sub(r'([A-Z])', r' \1', class_name).strip()
        name = name.replace("  ", " ")
        # Clean up common abbreviations
        name = name.replace("S V C", "SVC").replace("S V R", "SVR")
        name = name.replace("S G D", "SGD").replace("M L P", "MLP")
        name = name.replace("K Neighbors", "K-Neighbors")
        name = name.replace("L G B M", "LGBM").replace("X G B", "XGB")
        name = name.replace("N B", "NB")
        return name

    def get_classifiers(self) -> List[RegisteredModel]:
        """Return all registered classifiers, sorted by complexity."""
        complexity_order = {"Low": 0, "Medium": 1, "Medium-High": 2, "High": 3}
        models = [m for m in self._models.values() if m.task_type == "classifier"]
        return sorted(models, key=lambda m: complexity_order.get(m.complexity, 2))

    def get_regressors(self) -> List[RegisteredModel]:
        """Return all registered regressors, sorted by complexity."""
        complexity_order = {"Low": 0, "Medium": 1, "Medium-High": 2, "High": 3}
        models = [m for m in self._models.values() if m.task_type == "regressor"]
        return sorted(models, key=lambda m: complexity_order.get(m.complexity, 2))

    def get_model(self, class_name: str) -> Optional[RegisteredModel]:
        """Get a specific model by class name."""
        return self._models.get(class_name)

    def get_all(self) -> Dict[str, RegisteredModel]:
        """Return all registered models."""
        return self._models.copy()

    def get_models_for_task(self, task_type: str) -> List[RegisteredModel]:
        """Get models matching task: 'binary_classification', 'multiclass_classification', or 'regression'."""
        if "classification" in task_type:
            return self.get_classifiers()
        return self.get_regressors()

    def instantiate(self, class_name: str, **override_params) -> Any:
        """Create an instance of a registered model with optional parameter overrides."""
        model_info = self._models.get(class_name)
        if model_info is None:
            raise ValueError(f"Model '{class_name}' not found in registry.")
        params = {**model_info.default_params, **override_params}
        return model_info.estimator_class(**params)

    @property
    def classifier_count(self) -> int:
        return len(self.get_classifiers())

    @property
    def regressor_count(self) -> int:
        return len(self.get_regressors())

    @property
    def total_count(self) -> int:
        return len(self._models)


# Singleton instance
_registry = None

def get_registry() -> ModelRegistry:
    """Get the global model registry singleton."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry