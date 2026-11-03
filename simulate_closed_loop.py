import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.hydraulic_twin.config import REPORT_DIR
from src.hydraulic_twin.controller import control_decision
from src.hydraulic_twin.plant_model import HydraulicPlant
from src.hydraulic_twin.proportional_valve import ProportionalValve

DT = 0.1
SEGMENT_SECONDS = 20
DEMAND_SCHEDULE = [5.0, 10.0, 15.0]  # USER-SELECTED demand [L/min]

# In this demo the valve condition is CHOSEN BY YOU, not read from the ML model.
SCENARIOS = {
    "valve 100 (optimal)": 100,
    "valve 80 (severe lag)": 80,
    "valve 73 (near failure)": 73,
}


def run_scenario(condition: int):
    valve = ProportionalValve()
    plant = HydraulicPlant()
    log = {"t": [], "demand": [], "flow": [], "opening": [],
           "pressure": [], "mode": []}
    steps_per_segment = int(SEGMENT_SECONDS / DT)
    step_counter = 0

    for demand in DEMAND_SCHEDULE:
        for _ in range(steps_per_segment):
            decision = control_decision(
                desired_flow_lpm=demand,
                pressure_bar=plant.pressure_bar,
                sensor_valid=True,
                predicted_valve_condition=condition,
                current_opening_percent=valve.opening_percent,
            )
            opening = valve.command(
                decision["requested_opening_percent"], DT
            )
            flow = valve.estimate_flow(plant.pressure_bar)
            plant.step(flow, DT)

            log["t"].append(step_counter * DT)
            log["demand"].append(demand)
            log["flow"].append(flow)
            log["opening"].append(opening)
            log["pressure"].append(plant.pressure_bar)
            log["mode"].append(decision["mode"])
            step_counter += 1
    return log


results = {name: run_scenario(cond) for name, cond in SCENARIOS.items()}

print("ALL FLOWS, OPENINGS AND PRESSURES BELOW ARE SIMULATED.")
print("Valve condition is USER-SELECTED here (not an ML prediction).\n")
steps_per_segment = int(SEGMENT_SECONDS / DT)
for name, log in results.items():
    print(name)
    print(f"  {'demand':>7} {'flow':>7} {'error':>7} {'opening%':>9} "
          f"{'pressure':>9}  mode")
    for k in range(len(DEMAND_SCHEDULE)):
        i = (k + 1) * steps_per_segment - 1  # last step of the segment
        err = log["flow"][i] - log["demand"][i]
        print(f"  {log['demand'][i]:7.1f} {log['flow'][i]:7.2f} "
              f"{err:7.2f} {log['opening'][i]:9.1f} "
              f"{log['pressure'][i]:9.1f}  {log['mode'][i]}")
    print()

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
first = next(iter(results.values()))
axes[0].plot(first["t"], first["demand"], "k--", label="demand (user-selected)")
for name, log in results.items():
    axes[0].plot(log["t"], log["flow"], label=f"flow - {name}")
    axes[1].plot(log["t"], log["pressure"], label=name)
axes[0].set_ylabel("flow [L/min] (simulated)")
axes[0].legend(fontsize=8)
axes[1].set_ylabel("pressure [bar] (simulated)")
axes[1].set_xlabel("simulation time [s]")
axes[1].legend(fontsize=8)
fig.suptitle("SIMULATED closed-loop demo - assumed parameters, "
             "not dataset measurements")
fig.tight_layout()

REPORT_DIR.mkdir(exist_ok=True)
out_path = REPORT_DIR / "closed_loop_demo.png"
fig.savefig(out_path, dpi=120)
print("Saved:", out_path)