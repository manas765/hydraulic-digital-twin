"""Helpers for the ML experiments (model comparison, robustness checks)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .train_valve_model import VALVE_LABELS, make_model

COMPONENT_COLUMNS = [
    "cooler_condition",
    "pump_leakage",
    "accumulator_pressure",
]


def make_hist_gradient_boosting() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", HistGradientBoostingClassifier(
            class_weight="balanced", random_state=42,
        )),
    ])


def make_logistic_regression() -> Pipeline:
    # The scaler is fit inside each training fold only (no test leakage).
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=5000, class_weight="balanced",
        )),
    ])


def model_factories() -> dict:
    return {
        "Random forest (baseline)": make_model,
        "Hist. gradient boosting": make_hist_gradient_boosting,
        "Logistic regression (scaled)": make_logistic_regression,
    }


def cross_validate(model_factory, X, y, groups, n_splits=5, seed=42):
    """Same grouped, stratified folds as the baseline training script.
    Returns out-of-fold predictions and per-fold scores."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed)
    oof = np.zeros(len(y), dtype=int)
    fold_f1, fold_bal = [], []

    for train_idx, test_idx in cv.split(X, y, groups=groups):
        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        oof[test_idx] = pred
        fold_f1.append(f1_score(y.iloc[test_idx], pred,
                                average="macro", zero_division=0))
        fold_bal.append(balanced_accuracy_score(y.iloc[test_idx], pred))
    return oof, fold_f1, fold_bal


def class_recalls(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = {}
    for label in VALVE_LABELS:
        in_class = y_true == label
        recalls[f"recall_{label}"] = (
            float((y_pred[in_class] == label).mean())
            if in_class.any() else float("nan")
        )
    return recalls


def subgroup_report(results: pd.DataFrame, component: str) -> pd.DataFrame:
    """Valve-classification quality inside each label of another component.
    results needs: true_valve_condition, oof_predicted_valve_condition,
    and the component's label column."""
    rows = []
    for value, group in results.groupby(component):
        row = {
            "component": component,
            "label": int(value),
            "cycles": int(len(group)),
            "accuracy": float((
                group["true_valve_condition"]
                == group["oof_predicted_valve_condition"]
            ).mean()),
        }
        row.update(class_recalls(
            group["true_valve_condition"],
            group["oof_predicted_valve_condition"],
        ))
        rows.append(row)
    return pd.DataFrame(rows)