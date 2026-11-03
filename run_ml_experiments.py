import numpy as np
import pandas as pd

from src.hydraulic_twin.config import PROCESSED_DATA_DIR, REPORT_DIR
from src.hydraulic_twin.ml_experiments import (
    COMPONENT_COLUMNS,
    class_recalls,
    cross_validate,
    model_factories,
    subgroup_report,
)

BLOCK_SIZE = 50

features = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
targets = pd.read_csv(PROCESSED_DATA_DIR / "targets.csv")
REPORT_DIR.mkdir(exist_ok=True)

# Stable cycles only, exactly as in the baseline training run
mask = targets["stable_flag"] == 0
X = features.loc[mask].reset_index(drop=True)
y = targets.loc[mask, "valve_condition"].reset_index(drop=True)
cycle_ids = X.pop("cycle_id").to_numpy()
groups = cycle_ids // BLOCK_SIZE

print(f"Stable cycles: {len(X)}   Features: {X.shape[1]}   "
      f"Groups (blocks of {BLOCK_SIZE}): {len(np.unique(groups))}")

# ------------------------------------------------ model comparison
print("\n=== Model comparison (same grouped folds, default settings) ===")
rows = []
baseline_f1 = None
for name, factory in model_factories().items():
    print(f"Running {name} ...")
    oof, fold_f1, fold_bal = cross_validate(factory, X, y, groups)
    row = {
        "model": name,
        "macro_f1_mean": float(np.mean(fold_f1)),
        "macro_f1_std": float(np.std(fold_f1)),
        "balanced_acc_mean": float(np.mean(fold_bal)),
        "balanced_acc_std": float(np.std(fold_bal)),
    }
    row.update(class_recalls(y, oof))
    if baseline_f1 is None:
        baseline_f1 = np.array(fold_f1)
    row["fold_f1_diff_vs_baseline"] = float(
        np.mean(np.array(fold_f1) - baseline_f1)
    )
    rows.append(row)

comparison = pd.DataFrame(rows)
comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
print()
print(comparison.round(3).to_string(index=False))
print("\nCheck: the baseline row should match your saved cross-validation "
      "result (macro F1 0.863 +/- 0.070).")
print("Differences smaller than the fold-to-fold spread (the +/- values) "
      "are not evidence that one model is better.")

# ---------------------- robustness to other component conditions
print("\n=== Valve classification inside other components' conditions ===")
print("(baseline random forest, out-of-fold predictions, stable cycles)")
oof_path = REPORT_DIR / "valve_oof_predictions_stable_only.csv"
if not oof_path.exists():
    raise SystemExit("Run the training step first to create " + oof_path.name)

oof = pd.read_csv(oof_path)
joined = oof.merge(
    targets[COMPONENT_COLUMNS].reset_index(names="cycle_id"),
    on="cycle_id", how="left",
)

reports = [subgroup_report(joined, c) for c in COMPONENT_COLUMNS]
robustness = pd.concat(reports, ignore_index=True)
robustness.to_csv(REPORT_DIR / "robustness_by_component.csv", index=False)
print(robustness.round(3).to_string(index=False))
print("\nExperiments 5-7: if accuracy or recall drops in some subgroup, the "
      "valve model is being affected by that other component's condition.")
print("Saved: model_comparison.csv and robustness_by_component.csv in "
      "reports/")