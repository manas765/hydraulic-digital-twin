from src.hydraulic_twin.controller import control_decision


def decide(**overrides):
    args = dict(
        desired_flow_lpm=5,
        pressure_bar=100,
        sensor_valid=True,
        predicted_valve_condition=100,
        current_opening_percent=0,
    )
    args.update(overrides)
    return control_decision(**args)


def test_invalid_sensor_inhibits_control():
    assert decide(sensor_valid=False)["mode"] == "INHIBIT"


def test_critical_valve_condition_requests_review():
    result = decide(predicted_valve_condition=73)
    assert result["mode"] == "FAULT_REVIEW"
    assert result["requested_opening_percent"] == 0.0


def test_pressure_above_limit_enters_safe_state():
    assert decide(pressure_bar=141)["mode"] == "SAFE_SIMULATION_STATE"


def test_pressure_below_limit_enters_safe_state():
    assert decide(pressure_bar=4)["mode"] == "SAFE_SIMULATION_STATE"


def test_negative_demand_is_inhibited():
    assert decide(desired_flow_lpm=-1)["mode"] == "INHIBIT"


def test_demand_above_limit_is_inhibited():
    assert decide(desired_flow_lpm=19)["mode"] == "INHIBIT"


def test_unknown_valve_condition_is_inhibited():
    assert decide(predicted_valve_condition=55)["mode"] == "INHIBIT"


def test_normal_mode_converts_demand_to_opening():
    result = decide(desired_flow_lpm=10)
    assert result["mode"] == "NORMAL_SIMULATION"
    assert abs(result["requested_opening_percent"] - 50.0) < 1e-9


def test_small_lag_still_runs_normal_mode():
    assert decide(predicted_valve_condition=90)["mode"] == "NORMAL_SIMULATION"


def test_severe_lag_caps_opening_but_can_still_open():
    low = decide(predicted_valve_condition=80, desired_flow_lpm=2)
    high = decide(predicted_valve_condition=80, desired_flow_lpm=15)
    assert low["mode"] == "RESTRICTED_SIMULATION"
    assert abs(low["requested_opening_percent"] - 10.0) < 1e-9
    assert high["requested_opening_percent"] == 25.0