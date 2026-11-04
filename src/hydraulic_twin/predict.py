import json
import warnings
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_DIR, PROCESSED_DATA_DIR

MODEL_PATH = MODEL_DIR / "valve_health_pipeline.joblib"
FEATURES_PATH = MODEL_DIR / "valve_health_features.json"

CONDITION_MEANINGS = {
    100: "Optimal switching behaviour",
    90: "Small switching lag",
    80: "Severe switching lag",
    73: "Close to total failure",
}


def _train_from_processed_data():
    """Retrain the baseline model (stable cycles, same settings and seed as
    train_valve_model.py) from the processed feature table."""
    from .train_valve_model import make_model

    features_file = PROCESSED_DATA_DIR / "features.csv"
    targets_file = PROCESSED_DATA_DIR / "targets.csv"
    if not features_file.exists() or not targets_file.exists():
        raise FileNotFoundError(
            "No saved model and no processed data to train from. Run: "
            "python -m src.hydraulic_twin.preprocessing and "
            "python -m src.hydraulic_twin.train_valve_model"
        )

    X = pd.read_csv(features_file)
    targets = pd.read_csv(targets_file)
    stable = targets["stable_flag"] == 0
    X = X.loc[stable].drop(columns=["cycle_id"])
    y = targets.loc[stable, "valve_condition"]

    model = make_model()
    model.fit(X, y)
    return model, list(X.columns)


@lru_cache(maxsize=1)
def _load_model():
    try:
        model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH, encoding="utf-8") as f:
            feature_names = json.load(f)
        return model, feature_names
    except Exception as error:  # missing file, or saved by another version
        warnings.warn(
            f"Could not load the saved model ({type(error).__name__}); "
            "retraining it from the processed data."
        )

    model, feature_names = _train_from_processed_data()
    try:  # cache for next time; a read-only disk is not a problem
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        with open(FEATURES_PATH, "w", encoding="utf-8") as f:
            json.dump(feature_names, f, indent=2)
    except OSError:
        pass
    return model, feature_names


def predict_valve_health(feature_row: pd.DataFrame) -> dict:
    """Predict valve condition for ONE cycle (a 1-row feature DataFrame)."""
    model, feature_names = _load_model()

    if len(feature_row) != 1:
        raise ValueError("Pass exactly one cycle (a 1-row DataFrame)")

    missing = [c for c in feature_names if c not in feature_row.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} feature(s), e.g. {missing[:5]}"
        )

    X = feature_row[feature_names]  # right columns, right order

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Feature row contains NaN or infinite values")

    prediction = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]
    classes = model.named_steps["classifier"].classes_

    return {
        "source": "ML prediction (an estimate, not a measurement)",
        "predicted_condition": prediction,
        "condition_description": CONDITION_MEANINGS.get(prediction, "Unknown"),
        "class_probabilities": {
            str(int(label)): float(p) for label, p in zip(classes, probs)
        },
    }