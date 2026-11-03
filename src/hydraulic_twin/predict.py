import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_DIR

MODEL_PATH = MODEL_DIR / "valve_health_pipeline.joblib"
FEATURES_PATH = MODEL_DIR / "valve_health_features.json"

CONDITION_MEANINGS = {
    100: "Optimal switching behaviour",
    90: "Small switching lag",
    80: "Severe switching lag",
    73: "Close to total failure",
}


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Model files not found. Train first with: "
            "python -m src.hydraulic_twin.train_valve_model"
        )
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, encoding="utf-8") as f:
        feature_names = json.load(f)
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