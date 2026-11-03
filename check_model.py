import pandas as pd

from src.hydraulic_twin.config import PROCESSED_DATA_DIR
from src.hydraulic_twin.predict import predict_valve_health

X = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
y = pd.read_csv(PROCESSED_DATA_DIR / "targets.csv")

print("Valve condition vs stable flag (number of cycles):")
print(pd.crosstab(y["valve_condition"], y["stable_flag"]))

print("\nReload test. These cycles were in the final model's training data,")
print("so this checks the plumbing, NOT accuracy.")
for label in [100, 90, 80, 73]:
    match = y[(y["valve_condition"] == label) & (y["stable_flag"] == 0)]
    idx = int(match.index[0])
    result = predict_valve_health(X.iloc[[idx]])
    probs = {k: round(v, 2) for k, v in result["class_probabilities"].items()}
    print(f"cycle {idx}: true={label}  "
          f"predicted={result['predicted_condition']}  probs={probs}")