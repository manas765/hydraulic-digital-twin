import numpy as np
import pandas as pd

from .config import (
    RAW_DATA_DIR,
    SENSOR_RATES,
    PROFILE_COLUMNS,
    CYCLE_SECONDS,
    EXPECTED_CYCLES,
    EXPECTED_PROFILE_VALUES,
)


def read_sensor_file(sensor_name: str) -> pd.DataFrame:
    """Read one sensor file: rows = cycles, columns = samples in a cycle."""
    path = RAW_DATA_DIR / f"{sensor_name}.txt"

    if not path.exists():
        raise FileNotFoundError(f"Missing sensor file: {path}")

    data = pd.read_csv(path, sep=r"\s+", header=None)

    if data.empty:
        raise ValueError(f"{sensor_name} file is empty")

    data = data.apply(pd.to_numeric, errors="coerce")

    if data.isna().any().any():
        raise ValueError(f"{sensor_name} contains missing/non-numeric values")

    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError(f"{sensor_name} contains infinite values")

    expected_samples = CYCLE_SECONDS * SENSOR_RATES[sensor_name]

    if data.shape[1] != expected_samples:
        raise ValueError(
            f"{sensor_name}: expected {expected_samples} samples per cycle, "
            f"found {data.shape[1]}"
        )

    return data


def load_profile() -> pd.DataFrame:
    """Read profile.txt: one row per cycle, 5 condition labels."""
    path = RAW_DATA_DIR / "profile.txt"

    if not path.exists():
        raise FileNotFoundError(f"Missing profile file: {path}")

    profile = pd.read_csv(path, sep=r"\s+", header=None)

    if profile.shape[1] != len(PROFILE_COLUMNS):
        raise ValueError(
            f"Expected {len(PROFILE_COLUMNS)} profile columns, "
            f"found {profile.shape[1]}"
        )

    profile.columns = PROFILE_COLUMNS

    for column, allowed in EXPECTED_PROFILE_VALUES.items():
        found = set(profile[column].unique())
        unexpected = found - allowed
        if unexpected:
            raise ValueError(
                f"{column}: unexpected label values {sorted(unexpected)}; "
                f"documented values are {sorted(allowed)}"
            )

    return profile


def validate_dataset():
    profile = load_profile()

    if len(profile) != EXPECTED_CYCLES:
        raise ValueError(
            f"Expected {EXPECTED_CYCLES} profile rows, found {len(profile)}"
        )

    sensor_shapes = {}

    for sensor in SENSOR_RATES:
        print(f"Checking {sensor} ...")
        data = read_sensor_file(sensor)
        sensor_shapes[sensor] = data.shape

    bad = {s: shape for s, shape in sensor_shapes.items()
           if shape[0] != len(profile)}
    if bad:
        raise ValueError(
            f"Sensor/profile row mismatch: {bad}, profile={len(profile)}"
        )

    print("\nDataset validation passed.")
    print("Cycles in profile:", len(profile))

    print("\nSensor shapes (cycles, samples per cycle):")
    for sensor, shape in sensor_shapes.items():
        print(f"  {sensor:5s} {shape}  at {SENSOR_RATES[sensor]} Hz")

    print("\nValve-condition distribution:")
    print(profile["valve_condition"].value_counts().sort_index())

    print("\nStable flag distribution (0 = stable, 1 = may not be stable yet):")
    print(profile["stable_flag"].value_counts().sort_index())

    return profile