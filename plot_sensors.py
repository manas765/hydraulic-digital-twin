import matplotlib

matplotlib.use("Agg")  # draw to files, no pop-up windows

import matplotlib.pyplot as plt
import numpy as np

from src.hydraulic_twin.config import REPORT_DIR, SENSOR_RATES
from src.hydraulic_twin.data_loader import load_profile, read_sensor_file

# Units as listed in the dataset documentation
UNITS = {
    "PS1": "bar", "PS2": "bar", "PS3": "bar",
    "PS4": "bar", "PS5": "bar", "PS6": "bar",
    "EPS1": "W",
    "FS1": "L/min", "FS2": "L/min",
    "TS1": "degC", "TS2": "degC", "TS3": "degC", "TS4": "degC",
    "VS1": "mm/s", "CE": "%", "CP": "kW", "SE": "%",
}

VALVE_LABELS = [100, 90, 80, 73]
COMPARE_SENSORS = ["PS1", "EPS1", "FS1", "TS1"]
OVERVIEW_CYCLE = 0

profile = load_profile()
REPORT_DIR.mkdir(exist_ok=True)

# Pick the first stable cycle for each valve condition
chosen = {}
for label in VALVE_LABELS:
    match = profile[
        (profile["valve_condition"] == label) & (profile["stable_flag"] == 0)
    ]
    chosen[label] = int(match.index[0])

print("Overview cycle:", OVERVIEW_CYCLE)
print(profile.loc[[OVERVIEW_CYCLE]].to_string())
print("\nCycles chosen for valve comparison:")
for label, idx in chosen.items():
    print(f"valve {label}: cycle {idx}")
    print(profile.loc[[idx]].to_string(header=False))

rows_needed = sorted({OVERVIEW_CYCLE, *chosen.values()})

# Read one sensor at a time and keep only the cycles we need (saves memory)
signals = {}
for sensor in SENSOR_RATES:
    print(f"Reading {sensor} ...")
    data = read_sensor_file(sensor)
    signals[sensor] = {
        i: data.iloc[i].to_numpy(dtype=float) for i in rows_needed
    }
    del data


def time_axis(sensor, n_samples):
    return np.arange(n_samples) / SENSOR_RATES[sensor]


# Figure 1: all 17 sensors, one cycle
fig, axes = plt.subplots(6, 3, figsize=(14, 16))
axes = axes.ravel()
for ax, sensor in zip(axes, SENSOR_RATES):
    values = signals[sensor][OVERVIEW_CYCLE]
    ax.plot(time_axis(sensor, values.size), values, linewidth=0.8)
    ax.set_title(f"{sensor} [{UNITS[sensor]}] - {SENSOR_RATES[sensor]} Hz")
    ax.set_xlabel("time in cycle (s)")
for ax in axes[len(SENSOR_RATES):]:
    ax.set_visible(False)
fig.suptitle(f"Measured dataset values - cycle {OVERVIEW_CYCLE}", y=1.0)
fig.tight_layout()
overview_path = REPORT_DIR / "sensor_overview_cycle0.png"
fig.savefig(overview_path, dpi=120)
plt.close(fig)

# Figure 2: same sensors, one cycle per valve condition
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, sensor in zip(axes.ravel(), COMPARE_SENSORS):
    for label, idx in chosen.items():
        values = signals[sensor][idx]
        ax.plot(
            time_axis(sensor, values.size),
            values,
            linewidth=0.8,
            label=f"valve {label}% (cycle {idx})",
        )
    ax.set_title(f"{sensor} [{UNITS[sensor]}]")
    ax.set_xlabel("time in cycle (s)")
    ax.legend(fontsize=8)
fig.suptitle(
    "Measured dataset values - one cycle per valve condition\n"
    "(other components also differ between these cycles)"
)
fig.tight_layout()
compare_path = REPORT_DIR / "valve_condition_comparison.png"
fig.savefig(compare_path, dpi=120)
plt.close(fig)

print("\nSaved:", overview_path)
print("Saved:", compare_path)