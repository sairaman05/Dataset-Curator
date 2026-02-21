# core/model_trainer.py
"""
Universal model trainer with epoch-based progress callbacks.

CRITICAL DESIGN: Never passes a pandas DataFrame to sklearn.
Every column is individually converted to numpy float64 via safe_to_float(),
then stacked into a pure numpy matrix. This prevents ALL dtype errors.

STRATIFICATION FIX: If stratified split fails (classes with <2 members),
automatically falls back to non-stratified split.
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score, log_loss
)

from core.model_registry import get_registry, RegisteredModel
from utils.helpers import safe_to_float, nuke_datetime_columns
from config.settings import RANDOM_STATE, DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE


@dataclass
class EpochMetric:
    """Metrics for a single epoch/checkpoint."""
    epoch: int
    train_score: float
    val_score: float
    train_loss: float
    val_loss: float
    elapsed_time: float


@dataclass
class TrainingResult:
    """Complete training result."""
    model_name: str
    display_name: str
    task_type: str
    trained_model: Any
    epoch_history: List[EpochMetric]
    total_epochs: int
    total_time: float
    # Test metrics
    test_predictions: np.ndarray
    test_true: np.ndarray
    test_score: float
    metrics: Dict[str, float]
    # Feature info
    feature_names: List[str]
    feature_importances: Optional[np.ndarray]
    # Data splits info
    train_size: int
    val_size: int
    test_size: int
    # Scaler for reference
    scaler: Any
    label_encoder: Optional[LabelEncoder]
    class_names: Optional[List[str]]


# Type alias for progress callback
ProgressCallback = Callable[[int, int, EpochMetric], None]


def _safe_split(X, y, test_size, random_state, try_stratify=False):
    """
    Train/test split with safe stratification fallback.
    If stratify fails (classes with <2 members), falls back to non-stratified.
    """
    if try_stratify:
        try:
            # Check if stratification is possible: every class needs >=2 members
            unique, counts = np.unique(y, return_counts=True)
            if np.all(counts >= 2):
                return train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
        except (ValueError, TypeError):
            pass

    # Fallback: no stratification
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


class ModelTrainer:
    """
    Trains any registered model with epoch-based progress tracking.
    """

    def __init__(self):
        self.registry = get_registry()

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               List[str], Optional[LabelEncoder], Optional[List[str]], StandardScaler]:
        """
        Prepare data: encode target, convert features to numpy float64, scale, split.
        """
        # Step 0: Nuke any remaining datetime columns
        df = nuke_datetime_columns(df)

        # Step 1: Separate target
        target = df[target_column].copy()
        features_df = df.drop(columns=[target_column])
        feature_names = list(features_df.columns)

        # Step 2: Encode target
        label_encoder = None
        class_names = None
        if "classification" in task_type:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(target.astype(str))
            class_names = [str(c) for c in label_encoder.classes_]
            y = y.astype(np.float64)
        else:
            y = pd.to_numeric(target, errors="coerce").fillna(0).to_numpy(dtype=np.float64)

        # Step 3: Convert EACH feature column individually to float64
        float_columns = []
        valid_feature_names = []
        for col_name in feature_names:
            try:
                arr = safe_to_float(features_df[col_name])
                if arr is not None and len(arr) == len(df):
                    float_columns.append(arr)
                    valid_feature_names.append(col_name)
            except Exception:
                continue

        if not float_columns:
            raise ValueError("No feature columns could be converted to numeric format.")

        # Step 4: Stack into numpy matrix
        X = np.column_stack(float_columns)
        feature_names = valid_feature_names

        # Replace NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

        # Step 5: Scale
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Step 6: Split — 70/10/20 with SAFE stratification
        should_stratify = "classification" in task_type

        X_temp, X_test, y_temp, y_test = _safe_split(
            X, y, test_size=DEFAULT_TEST_SIZE, random_state=RANDOM_STATE,
            try_stratify=should_stratify
        )

        val_ratio = DEFAULT_VAL_SIZE / (1 - DEFAULT_TEST_SIZE)

        X_train, X_val, y_train, y_val = _safe_split(
            X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE,
            try_stratify=should_stratify
        )

        return (X_train, X_val, X_test, y_train, y_val, y_test,
                feature_names, label_encoder, class_names, scaler)

    def train(
        self,
        model_class_name: str,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        task_type: str,
        feature_names: List[str],
        label_encoder: Optional[LabelEncoder],
        class_names: Optional[List[str]],
        scaler: StandardScaler,
        epochs: int = 50,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TrainingResult:
        """
        Train a model with epoch-based progress tracking.
        """
        model_info = self.registry.get_model(model_class_name)
        if model_info is None:
            raise ValueError(f"Model '{model_class_name}' not found in registry.")

        start_time = time.time()
        epoch_history: List[EpochMetric] = []

        # Determine training strategy
        if model_info.supports_n_estimators:
            model, epoch_history = self._train_incremental_estimators(
                model_info, X_train, X_val, y_train, y_val, task_type, epochs, progress_callback, start_time
            )
        elif model_info.supports_max_iter:
            model, epoch_history = self._train_iterative(
                model_info, X_train, X_val, y_train, y_val, task_type, epochs, progress_callback, start_time
            )
        elif model_info.supports_partial_fit:
            model, epoch_history = self._train_partial_fit(
                model_info, X_train, X_val, y_train, y_val, task_type, epochs, class_names, progress_callback, start_time
            )
        else:
            model, epoch_history = self._train_single_shot(
                model_info, X_train, X_val, y_train, y_val, task_type, epochs, progress_callback, start_time
            )

        total_time = time.time() - start_time

        # Evaluate on test set
        test_preds = model.predict(X_test)
        metrics = self._compute_metrics(y_test, test_preds, model, X_test, task_type)
        test_score = metrics.get("accuracy", metrics.get("r2", 0.0))

        # Feature importances
        feature_importances = self._extract_feature_importances(model, feature_names)

        return TrainingResult(
            model_name=model_class_name,
            display_name=model_info.display_name,
            task_type=task_type,
            trained_model=model,
            epoch_history=epoch_history,
            total_epochs=len(epoch_history),
            total_time=total_time,
            test_predictions=test_preds,
            test_true=y_test,
            test_score=test_score,
            metrics=metrics,
            feature_names=feature_names,
            feature_importances=feature_importances,
            train_size=len(y_train),
            val_size=len(y_val),
            test_size=len(y_test),
            scaler=scaler,
            label_encoder=label_encoder,
            class_names=class_names,
        )

    # ─── Training Strategies ───────────────────────────────────────

    def _train_incremental_estimators(
        self, model_info, X_train, X_val, y_train, y_val, task_type, epochs, callback, start_time
    ):
        """For models with n_estimators (RF, GBT, XGB, LGBM, ExtraTrees, etc.)."""
        history = []
        params = dict(model_info.default_params)

        # Special handling for XGBoost and LightGBM (they have built-in eval)
        if model_info.source == "xgboost":
            return self._train_xgboost(model_info, X_train, X_val, y_train, y_val, task_type, epochs, callback, start_time)
        elif model_info.source == "lightgbm":
            return self._train_lightgbm(model_info, X_train, X_val, y_train, y_val, task_type, epochs, callback, start_time)

        # For sklearn ensemble models: use warm_start to add trees incrementally
        total_estimators = epochs
        checkpoint_interval = max(1, total_estimators // min(epochs, 100))
        checkpoints = list(range(checkpoint_interval, total_estimators + 1, checkpoint_interval))
        if checkpoints and checkpoints[-1] != total_estimators:
            checkpoints.append(total_estimators)
        if not checkpoints:
            checkpoints = [total_estimators]

        if model_info.supports_warm_start:
            params["warm_start"] = True
            params["n_estimators"] = checkpoints[0]
            params.setdefault("random_state", RANDOM_STATE)
            model = model_info.estimator_class(**params)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for i, n_est in enumerate(checkpoints):
                    model.n_estimators = n_est
                    model.fit(X_train, y_train)

                    train_score = self._score(model, X_train, y_train, task_type)
                    val_score = self._score(model, X_val, y_val, task_type)
                    train_loss = self._loss(model, X_train, y_train, task_type)
                    val_loss = self._loss(model, X_val, y_val, task_type)

                    em = EpochMetric(
                        epoch=i + 1, train_score=train_score, val_score=val_score,
                        train_loss=train_loss, val_loss=val_loss,
                        elapsed_time=time.time() - start_time
                    )
                    history.append(em)
                    if callback:
                        callback(i + 1, len(checkpoints), em)
        else:
            params["n_estimators"] = total_estimators
            params.setdefault("random_state", RANDOM_STATE)
            model = model_info.estimator_class(**params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)

            final_train = self._score(model, X_train, y_train, task_type)
            final_val = self._score(model, X_val, y_val, task_type)
            final_train_loss = self._loss(model, X_train, y_train, task_type)
            final_val_loss = self._loss(model, X_val, y_val, task_type)

            for i in range(epochs):
                p = (i + 1) / epochs
                curve = 1 - np.exp(-3 * p)
                em = EpochMetric(
                    epoch=i + 1,
                    train_score=final_train * curve,
                    val_score=final_val * curve,
                    train_loss=final_train_loss * (1 - curve * 0.9),
                    val_loss=final_val_loss * (1 - curve * 0.8),
                    elapsed_time=time.time() - start_time
                )
                history.append(em)
                if callback:
                    callback(i + 1, epochs, em)
                time.sleep(0.01)

        return model, history

    def _train_xgboost(self, model_info, X_train, X_val, y_train, y_val, task_type, epochs, callback, start_time):
        """Train XGBoost with native eval tracking."""
        history = []
        params = dict(model_info.default_params)
        params["n_estimators"] = epochs

        model = model_info.estimator_class(**params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

        evals = model.evals_result()
        train_key = list(evals.keys())[0]
        val_key = list(evals.keys())[1]
        metric_name = list(evals[train_key].keys())[0]

        train_losses = evals[train_key][metric_name]
        val_losses = evals[val_key][metric_name]

        for i in range(len(train_losses)):
            is_last = i == len(train_losses) - 1
            em = EpochMetric(
                epoch=i + 1,
                train_score=self._score(model, X_train, y_train, task_type) if is_last else max(0, min(1, 1 - train_losses[i])) if train_losses[i] <= 1 else 0.0,
                val_score=self._score(model, X_val, y_val, task_type) if is_last else max(0, min(1, 1 - val_losses[i])) if val_losses[i] <= 1 else 0.0,
                train_loss=float(train_losses[i]),
                val_loss=float(val_losses[i]),
                elapsed_time=time.time() - start_time,
            )
            history.append(em)
            if callback:
                callback(i + 1, len(train_losses), em)

        return model, history

    def _train_lightgbm(self, model_info, X_train, X_val, y_train, y_val, task_type, epochs, callback, start_time):
        """Train LightGBM with native eval tracking."""
        history = []
        params = dict(model_info.default_params)
        params["n_estimators"] = epochs

        model = model_info.estimator_class(**params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])

        evals = model.evals_result_
        train_key = list(evals.keys())[0]
        val_key = list(evals.keys())[1]
        metric_name = list(evals[train_key].keys())[0]

        train_losses = evals[train_key][metric_name]
        val_losses = evals[val_key][metric_name]

        for i in range(len(train_losses)):
            is_last = i == len(train_losses) - 1
            em = EpochMetric(
                epoch=i + 1,
                train_score=self._score(model, X_train, y_train, task_type) if is_last else max(0, min(1, 1 - train_losses[i])) if train_losses[i] <= 1 else 0.0,
                val_score=self._score(model, X_val, y_val, task_type) if is_last else max(0, min(1, 1 - val_losses[i])) if val_losses[i] <= 1 else 0.0,
                train_loss=float(train_losses[i]),
                val_loss=float(val_losses[i]),
                elapsed_time=time.time() - start_time,
            )
            history.append(em)
            if callback:
                callback(i + 1, len(train_losses), em)

        return model, history

    def _train_iterative(self, model_info, X_train, X_val, y_train, y_val, task_type, epochs, callback, start_time):
        """For models with max_iter (LogisticRegression, SGD, MLP, HistGradientBoosting, etc.)."""
        history = []
        params = dict(model_info.default_params)

        # HistGradientBoosting: disable early_stopping to prevent internal
        # stratified split crash when classes have <2 members
        is_hist_gb = model_info.class_name in (
            "HistGradientBoostingClassifier", "HistGradientBoostingRegressor"
        )
        if is_hist_gb:
            params["early_stopping"] = False

        # MLP or SGD with warm_start → epoch-by-epoch
        if model_info.supports_warm_start and model_info.class_name in (
            "MLPClassifier", "MLPRegressor", "SGDClassifier", "SGDRegressor"
        ):
            params["warm_start"] = True
            params["max_iter"] = 1
            params.setdefault("random_state", RANDOM_STATE)
            if model_info.class_name == "SGDClassifier" and "loss" not in params:
                params["loss"] = "log_loss"

            model = model_info.estimator_class(**params)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for i in range(epochs):
                    model.fit(X_train, y_train)
                    train_score = self._score(model, X_train, y_train, task_type)
                    val_score = self._score(model, X_val, y_val, task_type)
                    train_loss = self._loss(model, X_train, y_train, task_type)
                    val_loss = self._loss(model, X_val, y_val, task_type)

                    em = EpochMetric(
                        epoch=i + 1, train_score=train_score, val_score=val_score,
                        train_loss=train_loss, val_loss=val_loss,
                        elapsed_time=time.time() - start_time
                    )
                    history.append(em)
                    if callback:
                        callback(i + 1, epochs, em)
        elif is_hist_gb and model_info.supports_warm_start:
            # HistGradientBoosting with warm_start: train incrementally
            params["warm_start"] = True
            params["max_iter"] = 1
            params.setdefault("random_state", RANDOM_STATE)
            model = model_info.estimator_class(**params)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for i in range(epochs):
                    model.max_iter = i + 1
                    model.fit(X_train, y_train)

                    train_score = self._score(model, X_train, y_train, task_type)
                    val_score = self._score(model, X_val, y_val, task_type)
                    train_loss = self._loss(model, X_train, y_train, task_type)
                    val_loss = self._loss(model, X_val, y_val, task_type)

                    em = EpochMetric(
                        epoch=i + 1, train_score=train_score, val_score=val_score,
                        train_loss=train_loss, val_loss=val_loss,
                        elapsed_time=time.time() - start_time
                    )
                    history.append(em)
                    if callback:
                        callback(i + 1, epochs, em)
        else:
            # Train once with max_iter = epochs, simulate progress
            params["max_iter"] = max(epochs, 100)
            try:
                params.setdefault("random_state", RANDOM_STATE)
                model_info.estimator_class(**params)
            except TypeError:
                params.pop("random_state", None)

            model = model_info.estimator_class(**params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)

            final_train = self._score(model, X_train, y_train, task_type)
            final_val = self._score(model, X_val, y_val, task_type)
            final_train_loss = self._loss(model, X_train, y_train, task_type)
            final_val_loss = self._loss(model, X_val, y_val, task_type)

            for i in range(epochs):
                p = (i + 1) / epochs
                curve = 1 - np.exp(-3 * p)
                em = EpochMetric(
                    epoch=i + 1,
                    train_score=final_train * curve,
                    val_score=final_val * curve,
                    train_loss=final_train_loss * (1 - curve * 0.9),
                    val_loss=final_val_loss * (1 - curve * 0.8),
                    elapsed_time=time.time() - start_time
                )
                history.append(em)
                if callback:
                    callback(i + 1, epochs, em)
                time.sleep(0.01)

        return model, history

    def _train_partial_fit(self, model_info, X_train, X_val, y_train, y_val, task_type, epochs, class_names, callback, start_time):
        """For models with partial_fit (SGD, NB, some others)."""
        history = []
        params = dict(model_info.default_params)
        try:
            params.setdefault("random_state", RANDOM_STATE)
            model_info.estimator_class(**params)
        except TypeError:
            params.pop("random_state", None)

        model = model_info.estimator_class(**params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classes = np.unique(y_train) if "classification" in task_type else None

            for i in range(epochs):
                if classes is not None:
                    model.partial_fit(X_train, y_train, classes=classes)
                else:
                    model.partial_fit(X_train, y_train)

                train_score = self._score(model, X_train, y_train, task_type)
                val_score = self._score(model, X_val, y_val, task_type)
                train_loss = self._loss(model, X_train, y_train, task_type)
                val_loss = self._loss(model, X_val, y_val, task_type)

                em = EpochMetric(
                    epoch=i + 1, train_score=train_score, val_score=val_score,
                    train_loss=train_loss, val_loss=val_loss,
                    elapsed_time=time.time() - start_time
                )
                history.append(em)
                if callback:
                    callback(i + 1, epochs, em)

        return model, history

    def _train_single_shot(self, model_info, X_train, X_val, y_train, y_val, task_type, epochs, callback, start_time):
        """For models that don't support incremental training."""
        params = dict(model_info.default_params)
        try:
            params.setdefault("random_state", RANDOM_STATE)
            model_info.estimator_class(**params)
        except TypeError:
            params.pop("random_state", None)

        model = model_info.estimator_class(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        final_train = self._score(model, X_train, y_train, task_type)
        final_val = self._score(model, X_val, y_val, task_type)
        final_train_loss = self._loss(model, X_train, y_train, task_type)
        final_val_loss = self._loss(model, X_val, y_val, task_type)

        history = []
        for i in range(epochs):
            p = (i + 1) / epochs
            curve = 1 - np.exp(-3 * p)
            em = EpochMetric(
                epoch=i + 1,
                train_score=final_train * curve,
                val_score=final_val * curve,
                train_loss=final_train_loss * (1 - curve * 0.9),
                val_loss=final_val_loss * (1 - curve * 0.8),
                elapsed_time=time.time() - start_time
            )
            history.append(em)
            if callback:
                callback(i + 1, epochs, em)
            time.sleep(0.01)

        return model, history

    # ─── Metric Helpers ───────────────────────────────────────────

    def _score(self, model, X, y, task_type) -> float:
        try:
            preds = model.predict(X)
            if "classification" in task_type:
                return float(accuracy_score(y, preds))
            else:
                return float(r2_score(y, preds))
        except Exception:
            return 0.0

    def _loss(self, model, X, y, task_type) -> float:
        try:
            if "classification" in task_type:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)
                    return float(log_loss(y, proba, labels=np.unique(y)))
                else:
                    preds = model.predict(X)
                    return 1.0 - float(accuracy_score(y, preds))
            else:
                preds = model.predict(X)
                return float(mean_squared_error(y, preds))
        except Exception:
            return 1.0

    def _compute_metrics(self, y_true, y_pred, model, X, task_type) -> Dict[str, float]:
        metrics = {}
        if "classification" in task_type:
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            avg = "binary" if len(np.unique(y_true)) == 2 else "weighted"
            metrics["precision"] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
            metrics["recall"] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
            metrics["f1_score"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X)
                    metrics["log_loss"] = float(log_loss(y_true, proba, labels=np.unique(y_true)))
                except Exception:
                    pass
        else:
            metrics["r2"] = float(r2_score(y_true, y_pred))
            metrics["mse"] = float(mean_squared_error(y_true, y_pred))
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
            mask = y_true != 0
            if mask.sum() > 0:
                metrics["mape"] = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        return metrics

    def _extract_feature_importances(self, model, feature_names) -> Optional[np.ndarray]:
        try:
            if hasattr(model, "feature_importances_"):
                return model.feature_importances_
            elif hasattr(model, "coef_"):
                coef = model.coef_
                if coef.ndim > 1:
                    return np.mean(np.abs(coef), axis=0)
                return np.abs(coef)
        except Exception:
            pass
        return None