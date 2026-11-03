import re
from types import SimpleNamespace

import pytest

from src.hydraulic_twin import three_view as tv


def fake_state(predicted=80, alerts=None):
    sensors = {
        name: {"value": 1.2345, "unit": "u"}
        for name in tv.OBSERVED_SENSORS
    }
    probabilities = (
        {"73": 0.05, "80": 0.85, "90": 0.05, "100": 0.05}
        if predicted is not None else {}
    )
    return {
        "cycle_id": 7,
        "data_source": "UCI_DATASET_REPLAY",
        "predicted_valve_condition": {"value": predicted},
        "valve_condition_probabilities": {"value": probabilities},
        "observed_sensors": sensors,
        "dataset_labels": {"valve_condition": {"value": 80}},
        "control_mode": {"value": "NORMAL_SIMULATION", "reason": "demo"},
        "desired_flow_lpm": {"value": 8.0},
        "alerts": alerts or [],
    }


def fake_trace(n=5):
    return {key: [float(i) for i in range(n)] for key in tv.TRACE_KEYS}


PLANT = SimpleNamespace(supply_pressure_bar=150.0, relief_setting_bar=120.0)


def test_scene_data_has_expected_content():
    data = tv.build_scene_data(fake_state(), fake_trace(), PLANT)
    assert data["cycle_id"] == 7
    assert data["predicted_condition"] == 80
    assert data["predicted_description"] == "Severe lag"
    assert data["confidence"] == 0.85
    assert data["observed"]["PS1"]["value"] == 1.23
    assert set(data["trace"]) == set(tv.TRACE_KEYS)
    assert data["limits"]["relief_setting_bar"] == 120.0


def test_missing_prediction_is_handled():
    data = tv.build_scene_data(fake_state(predicted=None), fake_trace(), PLANT)
    assert data["predicted_condition"] is None
    assert data["confidence"] is None
    assert data["predicted_description"] == ""


def test_missing_trace_key_gives_a_clear_error():
    trace = fake_trace()
    del trace["simulated_relief_flow_lpm"]
    with pytest.raises(ValueError):
        tv.build_scene_data(fake_state(), trace, PLANT)


def test_rendered_page_has_no_placeholders_and_key_labels():
    html = tv.render_html(
        tv.build_scene_data(fake_state(), fake_trace(), PLANT)
    )
    assert "__SCENE_DATA__" not in html
    assert "__THREE_URL__" not in html
    assert tv.THREE_JS_URL in html
    for phrase in ("DATASET REPLAY", "ML PREDICTION", "SIMULATED",
                   "DATASET MEASUREMENT", "not simulated"):
        assert phrase in html


def test_text_in_data_cannot_break_out_of_the_script_tag():
    nasty = [{"level": "INFO", "source": "x",
              "message": "</script><b>boom</b>"}]
    html = tv.render_html(tv.build_scene_data(
        fake_state(alerts=[{"level": "INFO", "source": "simulation",
                            "message": "</script><b>boom</b>"}]),
        fake_trace(), PLANT))
    assert "</script><b>boom" not in html
    assert "\\u003c/script" in html
    assert nasty  # keeps the example readable


def test_every_element_id_used_by_the_script_exists_in_the_page():
    template = tv.TEMPLATE_PATH.read_text(encoding="utf-8")
    used = set(re.findall(r'\$\("([^"]+)"\)', template))
    defined = set(re.findall(r'id="([^"]+)"', template))
    assert used, "the page script should reference some elements"
    assert used <= defined, f"ids used but not defined: {used - defined}"