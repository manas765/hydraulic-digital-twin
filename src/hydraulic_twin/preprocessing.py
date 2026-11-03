import numpy as np
import pandas as pd

from .config import PROCESSED_DATA_DIR, SENSOR_RATES
from .data_loader import load_profile, read_sensor_file


def extract_sensor_features(values: np.ndarray, sensor: str) -> pd.DataFrame:
    """Summarize one sensor for ALL cycles at once.

    values: 2D array, shape (n_cycles, samples_per_cycle)
    returns: DataFrame with one row per cycle and 8 feature columns
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"{sensor}: expected a 2D array with >= 2 samples")
    if not np.isfinite(values).all():
        raise ValueError(f"Invalid values in {sensor}")

    # Time of each sample inside the cycle, in seconds
    time = np.arange(values.shape[1], dtype=float) / SENSOR_RATES[sensor]
    time_centered = time - time.mean()
    values_centered = values - values.mean(axis=1, keepdims=True)

    # Least-squares straight-line slope (units per second), per cycle
    slope = (values_centered * time_centered).sum(axis=1) / (
        time_centered ** 2
    ).sum()

    max_value = values.max(axis=1)
    min_value = values.min(axis=1)

    return pd.DataFrame(
        {
            f"{sensor}_mean": values.mean(axis=1),
            f"{sensor}_std": values.std(axis=1),
            f"{sensor}_min": min_value,
            f"{sensor}_max": max_value,
            f"{sensor}_median": np.median(values, axis=1),
            f"{sensor}_rms": np.sqrt((values ** 2).mean(axis=1)),
            f"{sensor}_range": max_value - min_value,
            f"{sensor}_slope_per_sec": slope,
        }
    )


def build_feature_table():
    profile = load_profile()
    per_sensor_features = []

    for sensor in SENSOR_RATES:
        print(f"Extracting features: {sensor} ...")
        sensor_data = read_sensor_file(sensor)

        if len(sensor_data) != len(profile):
            raise ValueError(
                f"{sensor}: {len(sensor_data)} rows but profile has "
                f"{len(profile)}"
            )

        features = extract_sensor_features(
            sensor_data.to_numpy(dtype=float), sensor
        )
        per_sensor_features.append(features)

    X = pd.concat(per_sensor_features, axis=1)

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Feature table contains NaN or infinite values")

    X.insert(0, "cycle_id", np.arange(len(X)))
    return X, profile


if __name__ == "__main__":
    X, y = build_feature_table()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    features_path = PROCESSED_DATA_DIR / "features.csv"
    targets_path = PROCESSED_DATA_DIR / "targets.csv"

    X.to_csv(features_path, index=False)
    y.to_csv(targets_path, index=False)

    print("\nFeatures:", X.shape)
    print("Targets:", y.shape)
    print("Saved:", features_path)
    print("Saved:", targets_path)