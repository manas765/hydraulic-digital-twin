import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.hydraulic_twin.config import REPORT_DIR
from src.hydraulic_twin.control_experiments import (
    run_demand_experiment,
    run_fault_experiment,
    run_relief_experiment,
)

REPORT_DIR.mkdir(exist_ok=True)
print("ALL RESULTS ARE SIMULATED with assumed parameters (see README).\n")

# ---------------------------------------------------- experiment 9
steps, summary, runs = run_demand_experiment()
steps.to_csv(REPORT_DIR / "control_step_metrics.csv", index=False)
summary.to_csv(REPORT_DIR / "control_policy_summary.csv", index=False)

print("=== Experiment 9: demand steps (5, 10, 6, 14, 8 L/min, 30 s each) ===")
print("settling = time to stay within 5 % (or 0.25 L/min) of the demand;")
print("NaN = never settled within the step. end_error = mean error over the")
print("last 5 s of the step (not always steady).\n")
print(steps.round(2).to_string(index=False))
print("\nWhole-run summary:")
print(summary.round(2).to_string(index=False))

# ---------------------------------------------------- experiment 10
faults, fault_runs = run_fault_experiment()
faults.to_csv(REPORT_DIR / "sensor_fault_metrics.csv", index=False)
print("\n=== Experiment 10: sensors invalid from t = 40 s to t = 70 s ===")
print(faults.round(2).to_string(index=False))

# ---------------------------------------------------- experiment 11
relief, relief_runs = run_relief_experiment()
relief.to_csv(REPORT_DIR / "relief_valve_experiment.csv", index=False)
print("\n=== Experiment 11: the same controller with and without a relief "
      "valve (demand 8 L/min) ===")
print(relief.round(2).to_string(index=False))

# ---------------------------------------------------- figure
fig, axes = plt.subplots(4, 1, figsize=(10, 13))
for label, result in runs.items():
    axes[0].plot(result["t"], result["flow"], label=label)
first = next(iter(runs.values()))
axes[0].plot(first["t"], first["demand"], "k--", label="demand")
axes[0].set_ylabel("flow [L/min]")
axes[0].set_title("Experiment 9: demand steps")
axes[0].legend(fontsize=8)
for label, result in runs.items():
    axes[1].plot(result["t"], result["pressure"], label=label)
axes[1].set_ylabel("pressure [bar]")
axes[1].set_title("Experiment 9: pressure")
axes[1].legend(fontsize=8)
axes[2].axvspan(40, 70, color="grey", alpha=0.2, label="sensors invalid",
                zorder=0)
for style, (label, result) in zip(("-", "--"), fault_runs.items()):
    axes[2].plot(result["t"], result["flow"], style, linewidth=2,
                 label=label, zorder=3)
axes[2].set_ylabel("flow [L/min]")
axes[2].set_title("Experiment 10: simulated sensor failure")
axes[2].legend(fontsize=8)
for label, result in relief_runs.items():
    axes[3].plot(result["t"], result["pressure"], label=label)
axes[3].axhline(140, color="orange", linestyle=":", label="controller limit")
axes[3].axhline(150, color="r", linestyle=":", label="hard pressure cap")
axes[3].set_ylabel("pressure [bar]")
axes[3].set_xlabel("simulated time [s]")
axes[3].set_title("Experiment 11: with and without a relief valve")
axes[3].legend(fontsize=8)
fig.suptitle("SIMULATED results with assumed parameters")
fig.tight_layout()
path = REPORT_DIR / "control_experiments.png"
fig.savefig(path, dpi=120)
print("\nSaved four CSV files and", path.name, "in reports/")
