import numpy as np

from src.hydraulic_twin import control_experiments as ce


def test_demand_based_control_tracks_a_reachable_demand():
    result = ce.simulate([(60, 8.0)], ce.demand_based(100))
    assert abs(result["flow"][-1] - 8.0) < 0.05
    assert len(result["t"]) == len(result["flow"]) == 600


def test_fixed_opening_ignores_the_demand():
    low = ce.simulate([(60, 5.0)], ce.fixed_opening(50.0))
    high = ce.simulate([(60, 10.0)], ce.fixed_opening(50.0))
    assert abs(low["flow"][-1] - 10.0) < 0.1
    assert abs(high["flow"][-1] - 10.0) < 0.1


def test_restricted_mode_caps_the_flow():
    result = ce.simulate([(60, 10.0)], ce.demand_based(80))
    assert abs(result["flow"][-1] - 5.0) < 0.1


def test_settling_time_and_overshoot_on_known_signals():
    t = np.arange(1, 101) * 0.1
    inside = np.full(100, 5.0)
    assert ce.settling_time(t, inside, 5.0, 0.0) == 0.0
    never = np.full(100, 9.0)
    assert ce.settling_time(t, never, 5.0, 0.0) is None
    late = np.where(t < 4.0, 0.0, 5.0)
    assert 3.9 < ce.settling_time(t, late, 5.0, 0.0) < 4.2
    assert ce.overshoot_percent([0, 4, 6, 5], 5.0, 0.0) == 20.0
    assert ce.overshoot_percent([0, 4, 5], 5.0, 0.0) == 0.0


def test_sensor_fault_closes_the_valve_and_recovers():
    table, _ = ce.run_fault_experiment()
    controller = table.iloc[0]
    assert controller["time_to_close_s"] < 2.0
    assert controller["peak_pressure_in_fault_bar"] < 140.0
    assert controller["recovery_time_s"] is not None
    assert controller["recovery_time_s"] < 5.0
    assert controller["final_mode"] == "NORMAL_SIMULATION"


def test_without_a_relief_valve_the_controller_latches_in_safe_state():
    table, _ = ce.run_relief_experiment()
    with_relief = table.iloc[0]
    without = table.iloc[1]
    assert with_relief["peak_pressure_bar"] < 140.0
    assert with_relief["final_mode"] == "NORMAL_SIMULATION"
    assert without["peak_pressure_bar"] == 150.0
    assert without["final_mode"] == "SAFE_SIMULATION_STATE"
    assert without["final_flow_lpm"] < 1e-3


def test_demand_experiment_ranks_the_policies_sensibly():
    _, summary, _ = ce.run_demand_experiment()
    errors = dict(zip(summary["policy"], summary["mean_abs_error_lpm"]))
    normal = errors["Demand-based, valve 100 (normal)"]
    restricted = errors["Demand-based, valve 80 (restricted)"]
    fixed = errors["Fixed 50 % opening (no control)"]
    assert normal < 0.5
    assert normal < fixed < restricted
