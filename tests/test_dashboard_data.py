import json

import numpy as np
import pandas as pd

from src.hydraulic_twin import dashboard_data as dd


def test_describe_label_known_and_unknown():
    assert dd.describe_label("valve_condition", 80) == "Severe lag"
    assert dd.describe_label("pump_leakage", 2) == "Severe leakage"
    assert dd.describe_label("valve_condition", 55) == "Unknown"


def test_example_cycles_picks_first_matching_cycles():
    targets = pd.DataFrame({
        "valve_condition": [100, 100, 90, 80, 73, 100],
        "stable_flag": [1, 0, 0, 0, 0, 0],
    })
    examples = dd.example_cycles(targets)
    assert examples["Valve 100 (stable)"] == 1
    assert examples["Valve 73 (stable)"] == 4
    assert examples["Unstable cycle"] == 0


def test_history_window_is_clamped_to_valid_range():
    assert dd.history_window(1000, 5, 100) == (0, 106)
    assert dd.history_window(1000, 995, 100) == (895, 1000)


def test_trace_frame_has_expected_columns_and_demand():
    trace = {
        "time_s": [0.1, 0.2],
        "requested_opening_percent": [40.0, 40.0],
        "simulated_opening_percent": [10.0, 18.0],
        "simulated_flow_lpm": [1.0, 2.0],
        "simulated_pressure_bar": [50.0, 52.0],
    }
    frame = dd.trace_frame(trace, desired_flow_lpm=8.0)
    for column in dd.OPENING_COLUMNS + dd.FLOW_COLUMNS + dd.PRESSURE_COLUMNS:
        assert column in frame.columns
    assert (frame["Demand [L/min] (user-selected)"] == 8.0).all()
    assert list(frame.index) == [0.1, 0.2]


def test_history_frame_leaves_uncovered_cycles_empty():
    targets = pd.DataFrame({"valve_condition": [100, 90, 80, 73]})
    oof = pd.DataFrame({
        "cycle_id": [1, 2],
        "oof_predicted_valve_condition": [90, 100],
    })
    frame = dd.history_frame(targets, oof, 0, 4)
    predicted = frame["Out-of-fold ML prediction [%]"]
    assert np.isnan(predicted.iloc[0]) and np.isnan(predicted.iloc[3])
    assert predicted.iloc[1] == 90 and predicted.iloc[2] == 100


def test_missing_cv_files_return_none(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "REPORT_DIR", tmp_path)
    assert dd.load_cv_results("stable_only") is None
    assert dd.load_oof_predictions("stable_only") is None


def test_confusion_and_report_frames_from_saved_results(tmp_path, monkeypatch):
    results = {
        "labels": [73, 80, 90, 100],
        "confusion_matrix": [[2, 0, 0, 0], [0, 2, 0, 0],
                             [0, 0, 1, 1], [0, 0, 0, 2]],
        "classification_report": {
            str(label): {"precision": 1.0, "recall": 1.0,
                         "f1-score": 1.0, "support": 2}
            for label in (73, 80, 90, 100)
        },
    }
    (tmp_path / "valve_cv_demo.json").write_text(json.dumps(results))
    monkeypatch.setattr(dd, "REPORT_DIR", tmp_path)

    loaded = dd.load_cv_results("demo")
    confusion = dd.confusion_frame(loaded)
    assert confusion.loc["True 90", "Predicted 100"] == 1
    assert list(dd.report_frame(loaded)["Valve class"]) == [73, 80, 90, 100]

    

def test_pretty_mode_and_condition_tone():
    assert dd.pretty_mode("RESTRICTED_SIMULATION") == "Restricted"
    assert dd.pretty_mode("SOMETHING_NEW") == "Something New"
    assert dd.condition_tone(73) == "critical"
    assert dd.condition_tone(None) == "none"


def test_robustness_findings_report_largest_gap_first():
    table = pd.DataFrame({
        "component": ["cooler_condition"] * 2 + ["pump_leakage"] * 2,
        "label": [3, 100, 0, 2],
        "cycles": [480, 489, 489, 480],
        "accuracy": [0.70, 0.95, 0.90, 0.85],
    })
    findings = dd.robustness_findings(table, gap_threshold=0.04)
    assert len(findings) == 2
    assert findings[0].startswith("Cooler condition")
    assert "0.70" in findings[0] and "label 3" in findings[0]
    assert dd.robustness_findings(table, gap_threshold=0.5) == []


def test_assumption_rows_come_from_the_code():
    rows = dd.assumption_rows()
    parameters = {(c, p): v for c, p, v, _ in rows}
    assert parameters[("Valve", "Maximum flow")] == 20.0
    assert parameters[("Plant", "Relief setting")] == 120.0
    assert parameters[("Controller", "Pressure limits")] == "5 to 140"