import pandas as pd

from src.hydraulic_twin.config import PROCESSED_DATA_DIR

X = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
y = pd.read_csv(PROCESSED_DATA_DIR / "targets.csv")

print("Features shape:", X.shape)
print("Targets shape:", y.shape)
print("Any missing values in features:", X.isna().any().any())
print("Any label columns leaked into features:",
      [c for c in y.columns if c in X.columns])

print("\nFirst 3 cycles, a few columns:")
print(X[["cycle_id", "PS1_mean", "FS1_mean", "TS1_mean",
         "EPS1_mean"]].head(3).to_string(index=False))

print("\nMean PS1_mean by valve condition (measured, summarized):")
print(X["PS1_mean"].groupby(y["valve_condition"]).mean().round(2))