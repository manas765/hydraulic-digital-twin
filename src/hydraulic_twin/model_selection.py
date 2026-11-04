"""Helpers for repeated cross-validation and sensor-group ablation."""
import numpy as np

from .config import SENSOR_RATES

ALL_SENSORS = list(SENSOR_RATES)
TEMPERATURE = ["TS1", "TS2", "TS3", "TS4"]
COOLING_EFFICIENCY = ["CE", "CP", "SE"]
HYDRAULIC_ONLY = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
                  "EPS1", "FS1", "FS2"]

# group name -> sensors that are KEPT
SENSOR_GROUPS = {
    "All 17 sensors": ALL_SENSORS,
    "Without temperature (TS1-TS4)": [
        s for s in ALL_SENSORS if s not in TEMPERATURE],
    "Without cooling/efficiency (CE, CP, SE)": [
        s for s in ALL_SENSORS if s not in COOLING_EFFICIENCY],
    "Without both": [
        s for s in ALL_SENSORS
        if s not in TEMPERATURE + COOLING_EFFICIENCY],
    "Pressure, motor power, flow only": HYDRAULIC_ONLY,
}

# Stated BEFORE running the experiment, so it cannot be bent to fit the
# results. A challenger replaces the baseline only if it wins on both.
MIN_SEED_WIN_FRACTION = 0.80
MIN_FOLD_WIN_FRACTION = 0.70


def columns_for_sensors(columns, sensors) -> list:
    """Feature columns (named like 'PS1_mean') belonging to the sensors."""
    keep = set(sensors)
    return [c for c in columns if c.split("_")[0] in keep]


def paired_comparison(f1_baseline, f1_challenger) -> dict:
    """Compare two models fold by fold (same folds, same order)."""
    base = np.asarray(f1_baseline, dtype=float)
    chal = np.asarray(f1_challenger, dtype=float)
    if base.shape != chal.shape or base.size == 0:
        raise ValueError("Need the same, non-empty set of folds")
    diff = chal - base
    return {
        "n_folds": int(diff.size),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std()),
        "fold_win_fraction": float((diff > 0).mean()),
    }


def seed_win_fraction(seed_means_baseline, seed_means_challenger) -> float:
    base = np.asarray(seed_means_baseline, dtype=float)
    chal = np.asarray(seed_means_challenger, dtype=float)
    return float((chal > base).mean())


def passes_decision_rule(seed_wins: float, fold_wins: float) -> bool:
    return (seed_wins >= MIN_SEED_WIN_FRACTION
            and fold_wins >= MIN_FOLD_WIN_FRACTION)


def subgroup_accuracy(y_true, y_pred, mask) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    return float((y_true[mask] == y_pred[mask]).mean())