import pytest

from src.hydraulic_twin.config import (
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    SENSOR_RATES,
)

REQUIRED_FILES = [
    PROCESSED_DATA_DIR / "features.csv",
    PROCESSED_DATA_DIR / "targets.csv",
    MODEL_DIR / "valve_health_pipeline.joblib",
    MODEL_DIR / "valve_health_features.json",
]

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in REQUIRED_FILES),
    reason="Run preprocessing and training first",
)


def make_twin():
    from src.hydraulic_twin.twin_state import DigitalTwin
    return DigitalTwin()


def first_cycle(twin, stable_flag):
    match = twin.targets[twin.targets["stable_flag"] == stable_flag]
    return int(match.index[0])


def test_every_value_carries_a_source():
    twin = make_twin()
    state = twin.advance_to_cycle(first_cycle(twin, 0))

    assert state["data_source"] == "UCI_DATASET_REPLAY"
    assert set(state["observed_sensors"]) == set(SENSOR_RATES)
    for item in state["observed_sensors"].values():
        assert item["source"] == "dataset"
    assert state["predicted_valve_condition"]["source"] == "ml_prediction"
    for key in (
        "simulated_valve_opening_percent",
        "simulated_valve_flow_lpm",
        "simulated_pressure_bar",
    ):
        assert state[key]["source"] == "simulation"
    assert state["requested_valve_opening_percent"]["source"] == (
        "controller_simulation"
    )


def test_unstable_cycle_raises_a_warning():
    twin = make_twin()
    state = twin.advance_to_cycle(first_cycle(twin, 1))
    messages = " ".join(a["message"] for a in state["alerts"])
    assert "not yet stable" in messages


def test_simulated_sensor_fault_inhibits_control():
    twin = make_twin()
    state = twin.advance_to_cycle(
        first_cycle(twin, 0), simulate_sensor_fault=True
    )
    assert state["control_mode"]["value"] == "INHIBIT"
    assert state["predicted_valve_condition"]["value"] is None
    assert state["requested_valve_opening_percent"]["value"] == 0.0


def test_replay_does_not_change_observed_values():
    twin = make_twin()
    cycle = first_cycle(twin, 0)
    first = twin.advance_to_cycle(cycle)
    second = twin.advance_to_cycle(cycle)

    assert first["observed_sensors"] == second["observed_sensors"]
    assert second["simulation_time_s"] == 2 * first["simulation_time_s"]


def test_invalid_cycle_id_is_rejected():
    twin = make_twin()
    with pytest.raises(IndexError):
        twin.advance_to_cycle(-1)
    with pytest.raises(IndexError):
        twin.advance_to_cycle(twin.n_cycles)