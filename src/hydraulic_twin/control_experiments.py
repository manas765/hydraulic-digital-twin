"""Closed-loop experiments with the simulated valve, plant and controller.

Everything here is SIMULATED with assumed parameters. No dataset is needed.
"""
import numpy as np
import pandas as pd

from .controller import control_decision
from .plant_model import HydraulicPlant
from .proportional_valve import ProportionalValve

DT_SECONDS = 0.1


def demand_based(condition: int):
    """The implemented controller, with a fixed predicted valve condition."""
    def policy(demand, plant, valve, sensor_valid):
        decision = control_decision(
            demand, plant.pressure_bar, sensor_valid,
            condition if sensor_valid else -1, valve.opening_percent,
        )
        return decision["requested_opening_percent"], decision["mode"]
    return policy


def fixed_opening(percent: float):
    """No control logic: the opening never changes."""
    def policy(demand, plant, valve, sensor_valid):
        return percent, "FIXED"
    return policy


def simulate(schedule, policy, fault_windows=(), plant=None, valve=None,
             dt=DT_SECONDS) -> dict:
    """schedule: list of (duration_s, demand_lpm).
    fault_windows: list of (start_s, end_s) with simulated invalid sensors."""
    plant = plant or HydraulicPlant()
    valve = valve or ProportionalValve()
    keys = ("t", "demand", "requested", "opening", "flow", "pressure",
            "relief", "mode", "sensor_valid")
    out = {k: [] for k in keys}
    t = 0.0
    for duration, demand in schedule:
        for _ in range(int(round(duration / dt))):
            valid = not any(a <= t < b for a, b in fault_windows)
            requested, mode = policy(demand, plant, valve, valid)
            valve.command(requested, dt)
            flow = valve.estimate_flow(plant.pressure_bar)
            plant.step(flow, dt)
            t += dt
            for k, v in zip(keys, (t, demand, requested,
                                   valve.opening_percent, flow,
                                   plant.pressure_bar, plant.relief_flow_lpm,
                                   mode, valid)):
                out[k].append(v)
    return {k: (v if k == "mode" else np.array(v)) for k, v in out.items()}


def tolerance_band(target: float) -> float:
    """Flow is 'on target' within 5 % of the demand or 0.25 L/min."""
    return max(0.05 * abs(target), 0.25)


def settling_time(t, flow, target, start, dt=DT_SECONDS):
    """Seconds after `start` until flow stays within the band to the end of
    the segment. None if it is outside the band at the end."""
    inside = np.abs(np.asarray(flow) - target) <= tolerance_band(target)
    if not inside[-1]:
        return None
    outside = np.where(~inside)[0]
    if len(outside) == 0:
        return 0.0
    return float(np.asarray(t)[outside[-1]] - start + dt)


def overshoot_percent(flow, target, before) -> float:
    """Peak excursion beyond the target in the direction of the step, as a
    percentage of the step size."""
    flow = np.asarray(flow)
    step = target - before
    if abs(step) < 1e-9:
        return 0.0
    excess = (flow.max() - target) if step > 0 else (target - flow.min())
    return float(max(0.0, excess) / abs(step) * 100.0)


def demand_step_table(result, schedule, label, relief_setting_bar=120.0):
    """One row per demand step for one policy."""
    rows, start, before = [], 0.0, 0.0
    t, flow = result["t"], result["flow"]
    for duration, demand in schedule:
        end = start + duration
        mask = (t > start) & (t <= end)
        seg_t, seg_flow = t[mask], flow[mask]
        tail = seg_flow[-int(5 / DT_SECONDS):]
        rows.append({
            "policy": label,
            "step_start_s": start,
            "demand_lpm": demand,
            "settling_time_s": settling_time(seg_t, seg_flow, demand, start),
            "end_error_lpm": float(tail.mean() - demand),
            "overshoot_pct": overshoot_percent(seg_flow, demand, before),
            "iae_lpm_s": float(np.abs(seg_flow - demand).sum() * DT_SECONDS),
            "peak_pressure_bar": float(result["pressure"][mask].max()),
        })
        before, start = float(seg_flow[-1]), end
    return pd.DataFrame(rows)


def run_demand_experiment(schedule=None):
    """Experiment 9: demand steps under three policies."""
    schedule = schedule or [(30, 5.0), (30, 10.0), (30, 6.0),
                            (30, 14.0), (30, 8.0)]
    policies = {
        "Demand-based, valve 100 (normal)": demand_based(100),
        "Demand-based, valve 80 (restricted)": demand_based(80),
        "Fixed 50 % opening (no control)": fixed_opening(50.0),
    }
    tables, summaries, runs = [], [], {}
    for label, policy in policies.items():
        result = simulate(schedule, policy)
        runs[label] = result
        tables.append(demand_step_table(result, schedule, label))
        opening = result["requested"]
        summaries.append({
            "policy": label,
            "mean_abs_error_lpm": float(np.abs(
                result["flow"] - result["demand"]).mean()),
            "peak_pressure_bar": float(result["pressure"].max()),
            "seconds_above_relief_setting": float(
                (result["pressure"] > 120.0).sum() * DT_SECONDS),
            "requested_opening_travel_pct": float(
                np.abs(np.diff(opening)).sum()),
        })
    return pd.concat(tables, ignore_index=True), pd.DataFrame(summaries), runs


def fault_metrics(result, fault_start, fault_end, demand, label):
    t = result["t"]
    after = t > fault_start
    closed = np.where(after & (result["opening"] <= 5.0))[0]
    low_flow = np.where(after & (result["flow"] < 0.5))[0]
    in_fault = (t > fault_start) & (t <= fault_end)

    recovered = None
    post = t > fault_end
    inside = np.abs(result["flow"][post] - demand) <= tolerance_band(demand)
    if inside.size and inside[-1]:
        out = np.where(~inside)[0]
        recovered = 0.0 if len(out) == 0 else float(
            t[post][out[-1]] - fault_end + DT_SECONDS)
    return {
        "scenario": label,
        "time_to_close_s": (None if len(closed) == 0
                            else float(t[closed[0]] - fault_start)),
        "time_flow_below_0.5_s": (None if len(low_flow) == 0
                                  else float(t[low_flow[0]] - fault_start)),
        "peak_pressure_in_fault_bar": float(
            result["pressure"][in_fault].max()),
        "pressure_at_fault_end_bar": float(result["pressure"][in_fault][-1]),
        "recovery_time_s": recovered,
        "final_mode": result["mode"][-1],
    }


def run_fault_experiment(total_s=120.0, demand=8.0, fault=(40.0, 70.0)):
    """Experiment 10: sensors become invalid for 30 s, then recover."""
    schedule = [(total_s, demand)]
    scenarios = {
        "Demand-based controller":
            (demand_based(100), HydraulicPlant()),
        "Fixed 40 % opening (ignores sensor validity)":
            (fixed_opening(40.0), HydraulicPlant()),
    }
    rows, runs = [], {}
    for label, (policy, plant) in scenarios.items():
        result = simulate(schedule, policy, fault_windows=[fault],
                          plant=plant)
        runs[label] = result
        rows.append(fault_metrics(result, fault[0], fault[1], demand, label))
    return pd.DataFrame(rows), runs


def run_relief_experiment(total_s=120.0, demand=8.0, limit_bar=140.0):
    """Experiment 11 (supplementary): why the simulated plant needs a relief
    path. Same controller and demand, with and without the relief valve."""
    scenarios = {
        "Relief valve present": HydraulicPlant(),
        "No relief valve": HydraulicPlant(relief_gain_lpm_per_bar=0.0),
    }
    rows, runs = [], {}
    for label, plant in scenarios.items():
        result = simulate([(total_s, demand)], demand_based(100), plant=plant)
        runs[label] = result
        over = np.where(result["pressure"] > limit_bar)[0]
        rows.append({
            "scenario": label,
            "first_time_above_140_bar_s": (
                None if len(over) == 0 else float(result["t"][over[0]])),
            "peak_pressure_bar": float(result["pressure"].max()),
            "final_mode": result["mode"][-1],
            "final_opening_pct": float(result["opening"][-1]),
            "final_flow_lpm": float(result["flow"][-1]),
        })
    return pd.DataFrame(rows), runs
