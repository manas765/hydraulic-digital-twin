"""Builds the self-contained 3D schematic page (three.js) from a twin state.

The page animates ONLY simulated values (valve opening, flow, pressure).
Observed dataset values and the ML prediction are shown as labelled text.
"""
import json
from pathlib import Path

from .dashboard_data import SOURCE_LABELS, describe_label

TEMPLATE_PATH = Path(__file__).resolve().parent / "assets" / (
    "twin_3d_template.html"
)
# Pinned three.js build (global/UMD script, no extra files needed)
THREE_JS_URL = (
    "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
)

OBSERVED_SENSORS = ("PS1", "FS1", "TS1", "EPS1")
TRACE_KEYS = (
    "time_s",
    "requested_opening_percent",
    "simulated_opening_percent",
    "simulated_flow_lpm",
    "simulated_pressure_bar",
    "simulated_relief_flow_lpm",
)


def build_scene_data(state: dict, trace: dict, plant) -> dict:
    missing = [k for k in TRACE_KEYS if k not in trace]
    if missing:
        raise ValueError(
            "The simulation trace is missing " + ", ".join(missing)
            + ". Apply the twin_state.py edits for the 3D step."
        )

    predicted = state["predicted_valve_condition"]["value"]
    probabilities = state["valve_condition_probabilities"]["value"]

    observed = {}
    for name in OBSERVED_SENSORS:
        item = state["observed_sensors"][name]
        observed[name] = {
            "value": round(float(item["value"]), 2),
            "unit": item["unit"],
        }

    return {
        "cycle_id": int(state["cycle_id"]),
        "data_source": state["data_source"],
        "predicted_condition": predicted,
        "predicted_description": (
            describe_label("valve_condition", predicted)
            if predicted is not None else ""
        ),
        "confidence": (
            float(max(probabilities.values())) if probabilities else None
        ),
        "dataset_valve_label": int(
            state["dataset_labels"]["valve_condition"]["value"]
        ),
        "control_mode": state["control_mode"]["value"],
        "control_reason": state["control_mode"]["reason"],
        "desired_flow_lpm": float(state["desired_flow_lpm"]["value"]),
        "observed": observed,
        "alerts": [
            {
                "level": a["level"],
                "source": SOURCE_LABELS.get(a["source"], a["source"]),
                "message": a["message"],
            }
            for a in state["alerts"]
        ],
        "limits": {
            "supply_pressure_bar": float(plant.supply_pressure_bar),
            "relief_setting_bar": float(plant.relief_setting_bar),
        },
        "trace": {
            key: [round(float(v), 3) for v in trace[key]]
            for key in TRACE_KEYS
        },
    }


def render_html(scene_data: dict) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    # "<" is escaped so no text in the data can close the script tag
    payload = json.dumps(scene_data).replace("<", "\\u003c")
    return (
        template
        .replace("__THREE_URL__", THREE_JS_URL)
        .replace("__SCENE_DATA__", payload)
    )


def write_html(path, scene_data: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(scene_data), encoding="utf-8")
    return path