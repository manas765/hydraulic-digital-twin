from pathlib import Path

# This file lives at: <project>/src/hydraulic_twin/config.py
# parents[0] = hydraulic_twin, parents[1] = src, parents[2] = <project>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

CYCLE_SECONDS = 60
EXPECTED_CYCLES = 2205

# Sampling rate (Hz) of each sensor, from the dataset documentation
SENSOR_RATES = {
    "PS1": 100, "PS2": 100, "PS3": 100,
    "PS4": 100, "PS5": 100, "PS6": 100,
    "EPS1": 100,
    "FS1": 10, "FS2": 10,
    "TS1": 1, "TS2": 1, "TS3": 1, "TS4": 1,
    "VS1": 1, "CE": 1, "CP": 1, "SE": 1,
}

# Units as listed in the dataset documentation
SENSOR_UNITS = {
    "PS1": "bar", "PS2": "bar", "PS3": "bar",
    "PS4": "bar", "PS5": "bar", "PS6": "bar",
    "EPS1": "W",
    "FS1": "L/min", "FS2": "L/min",
    "TS1": "degC", "TS2": "degC", "TS3": "degC", "TS4": "degC",
    "VS1": "mm/s", "CE": "%", "CP": "kW", "SE": "%",
}

PROFILE_COLUMNS = [
    "cooler_condition",
    "valve_condition",
    "pump_leakage",
    "accumulator_pressure",
    "stable_flag",
]

# Label values documented for each profile column
EXPECTED_PROFILE_VALUES = {
    "cooler_condition": {3, 20, 100},
    "valve_condition": {73, 80, 90, 100},
    "pump_leakage": {0, 1, 2},
    "accumulator_pressure": {90, 100, 115, 130},
    "stable_flag": {0, 1},
}