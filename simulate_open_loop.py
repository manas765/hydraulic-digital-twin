import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.hydraulic_twin.config import REPORT_DIR
from src.hydraulic_twin.plant_model import HydraulicPlant
from src.hydraulic_twin.proportional_valve import ProportionalValve

DT = 0.1
SEGMENT_SECONDS = 30
REQUEST_SCHEDULE = [70.0, 30.0, 100.0]  # USER-SELECTED requests, not dataset
PRINT_EVERY_SECONDS = 5

valve = ProportionalValve()
plant = HydraulicPlant()

times, requested_log, opening_log, flow_log, pressure_log = [], [], [], [], []
steps_per_segment = int(SEGMENT_SECONDS / DT)
print_every_steps = int(PRINT_EVERY_SECONDS / DT)
step_counter = 0

print("ALL VALUES BELOW ARE SIMULATED (assumed parameters).")
print(f"{'t(s)':>6} {'request%':>9} {'opening%':>9} "
      f"{'flow L/min':>11} {'pressure bar':>13}")

for requested in REQUEST_SCHEDULE:
    for _ in range(steps_per_segment):
        t = step_counter * DT
        opening = valve.command(requested, DT)
        flow = valve.estimate_flow(plant.pressure_bar)
        pressure = plant.step(flow, DT)

        times.append(t)
        requested_log.append(requested)
        opening_log.append(opening)
        flow_log.append(flow)
        pressure_log.append(pressure)

        if step_counter % print_every_steps == 0:
            print(f"{t:6.1f} {requested:9.1f} {opening:9.1f} "
                  f"{flow:11.2f} {pressure:13.1f}")
        step_counter += 1

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
axes[0].plot(times, requested_log, label="requested (user-selected)")
axes[0].plot(times, opening_log, label="valve opening (simulated)")
axes[0].set_ylabel("opening [%]")
axes[0].legend()

axes[1].plot(times, flow_log, color="tab:green")
axes[1].set_ylabel("valve flow [L/min] (simulated)")

axes[2].plot(times, pressure_log, color="tab:red")
axes[2].set_ylabel("pressure [bar] (simulated)")
axes[2].set_xlabel("simulation time [s]")

fig.suptitle("SIMULATED values with assumed parameters - not dataset "
             "measurements")
fig.tight_layout()

REPORT_DIR.mkdir(exist_ok=True)
out_path = REPORT_DIR / "plant_open_loop_demo.png"
fig.savefig(out_path, dpi=120)
print("\nSaved:", out_path)