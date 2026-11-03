"""Plain helper functions for the dashboard (no Streamlit imports here,
so they can be tested on their own)."""
import json

import numpy as np
import pandas as pd

from .config import REPORT_DIR

# Wording required by the project plan for every displayed value
SOURCE_LABELS = {
    "dataset": "DATASET MEASUREMENT",
    "ml_prediction": "ML PREDICTION",
    "simulation": "SIMULATED VALUE",
    "controller_simulation": "SIMULATED VALUE (controller)",
    "user_setting": "USER-SELECTED DEMAND",
}

COMPONENT_TITLES = {
    "cooler_condition": "Cooler condition [%]",
    "valve_condition": "Valve condition [%]",
    "pump_leakage": "Internal pump leakage",
    "accumulator_pressure": "Accumulator pressure [bar]",
}

# Meaning of each label value (verify against data/raw/description.txt)
COMPONENT_MEANINGS = {
    "cooler_condition": {
        3: "Close to total failure",
        20: "Reduced efficiency",
        100: "Full efficiency",
    },
    "valve_condition": {
        100: "Optimal switching behaviour",
        90: "Small lag",
        80: "Severe lag",
        73: "Close to total failure",
    },
    "pump_leakage": {
        0: "No leakage",
        1: "Weak leakage",
        2: "Severe leakage",
    },
    "accumulator_pressure": {
        130: "Optimal pressure",
        115: "Slightly reduced pressure",
        100: "Severely reduced pressure",
        90: "Close to total failure",
    },
}

OPENING_COLUMNS = [
    "Requested opening [%] (controller)",
    "Simulated opening [%]",
]
FLOW_COLUMNS = [
    "Simulated flow [L/min]",
    "Demand [L/min] (user-selected)",
]
PRESSURE_COLUMNS = ["Simulated pressure [bar]"]


def describe_label(component: str, value) -> str:
    return COMPONENT_MEANINGS.get(component, {}).get(int(value), "Unknown")


def example_cycles(targets: pd.DataFrame) -> dict:
    """First cycle of each demo scenario (stable cycle per valve class,
    plus the first cycle flagged as unstable)."""

    def first(valve, stable):
        match = targets[
            (targets["valve_condition"] == valve)
            & (targets["stable_flag"] == stable)
        ]
        return int(match.index[0]) if len(match) else None

    candidates = {
        "Valve 100 (stable)": first(100, 0),
        "Valve 90 (stable)": first(90, 0),
        "Valve 80 (stable)": first(80, 0),
        "Valve 73 (stable)": first(73, 0),
        "Unstable cycle": first(100, 1),
    }
    return {k: v for k, v in candidates.items() if v is not None}


def sensor_table(state: dict) -> pd.DataFrame:
    rows = [
        {
            "Sensor": name,
            "Cycle mean": item["value"],
            "Unit": item["unit"],
            "Source": SOURCE_LABELS[item["source"]],
        }
        for name, item in state["observed_sensors"].items()
    ]
    return pd.DataFrame(rows)


def trace_frame(trace: dict, desired_flow_lpm: float) -> pd.DataFrame:
    frame = pd.DataFrame({
        "Requested opening [%] (controller)":
            trace["requested_opening_percent"],
        "Simulated opening [%]": trace["simulated_opening_percent"],
        "Simulated flow [L/min]": trace["simulated_flow_lpm"],
        "Demand [L/min] (user-selected)": desired_flow_lpm,
        "Simulated pressure [bar]": trace["simulated_pressure_bar"],
    }, index=pd.Index(trace["time_s"], name="Simulated time in cycle [s]"))
    return frame


def history_window(n_cycles: int, cycle_id: int, half_width: int = 100):
    start = max(0, cycle_id - half_width)
    stop = min(n_cycles, cycle_id + half_width + 1)
    return start, stop


def sensor_history(features: pd.DataFrame, sensor: str, unit: str,
                   start: int, stop: int) -> pd.DataFrame:
    column = f"{sensor}_mean"
    frame = features[column].iloc[start:stop].to_frame(
        name=f"{sensor} cycle mean [{unit}]"
    )
    frame.index = features["cycle_id"].iloc[start:stop].to_numpy()
    frame.index.name = "Dataset cycle"
    return frame


def history_frame(targets: pd.DataFrame, oof: pd.DataFrame,
                  start: int, stop: int) -> pd.DataFrame:
    """Dataset valve label next to the out-of-fold ML prediction.
    Cycles the cross-validation run did not cover stay empty (NaN)."""
    cycle_ids = np.arange(start, stop)
    predictions = oof.set_index("cycle_id")[
        "oof_predicted_valve_condition"
    ].reindex(cycle_ids)

    return pd.DataFrame({
        "Dataset valve label [%]":
            targets["valve_condition"].iloc[start:stop].to_numpy(),
        "Out-of-fold ML prediction [%]": predictions.to_numpy(),
    }, index=pd.Index(cycle_ids, name="Dataset cycle"))


def load_cv_results(run_name: str):
    path = REPORT_DIR / f"valve_cv_{run_name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_oof_predictions(run_name: str):
    path = REPORT_DIR / f"valve_oof_predictions_{run_name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def confusion_frame(results: dict) -> pd.DataFrame:
    labels = results["labels"]
    return pd.DataFrame(
        results["confusion_matrix"],
        index=[f"True {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels],
    )


def report_frame(results: dict) -> pd.DataFrame:
    report = results["classification_report"]
    rows = []
    for label in results["labels"]:
        item = report[str(label)]
        rows.append({
            "Valve class": label,
            "Precision": item["precision"],
            "Recall": item["recall"],
            "F1": item["f1-score"],
            "Cycles": int(item["support"]),
        })
    return pd.DataFrame(rows)