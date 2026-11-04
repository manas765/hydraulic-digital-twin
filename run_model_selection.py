import argparse

import numpy as np
import pandas as pd

from src.hydraulic_twin.config import PROCESSED_DATA_DIR, REPORT_DIR
from src.hydraulic_twin.ml_experiments import (
    cross_validate,
    make_hist_gradient_boosting,
    make_logistic_regression,
)
from src.hydraulic_twin.model_selection import (
    MIN_FOLD_WIN_FRACTION,
    MIN_SEED_WIN_FRACTION,
    SENSOR_GROUPS,
    columns_for_sensors,
    paired_comparison,
    passes_decision_rule,
    seed_win_fraction,
    subgroup_accuracy,
)
from src.hydraulic_twin.train_valve_model import make_model

BLOCK_SIZE = 50

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=10,
                    help="number of different fold assignments (default 10)")
parser.add_argument("--ablation-seeds", type=int, default=3,
                    help="seeds used for the sensor-group ablation")
parser.add_argument("--skip-ablation", action="store_true")
args = parser.parse_args()

features = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
targets = pd.read_csv(PROCESSED_DATA_DIR / "targets.csv")
REPORT_DIR.mkdir(exist_ok=True)

mask = targets["stable_flag"] == 0
X = features.loc[mask].reset_index(drop=True)
y = targets.loc[mask, "valve_condition"].reset_index(drop=True)
cooler = targets.loc[mask, "cooler_condition"].reset_index(drop=True)
cycle_ids = X.pop("cycle_id").to_numpy()
groups = cycle_ids // BLOCK_SIZE
cooler_failing = (cooler == 3).to_numpy()

print(f"Stable cycles: {len(X)}   Features: {X.shape[1]}")
print("\nDecision rule, fixed before running: a challenger replaces the "
      f"baseline only if it has the higher mean macro F1 in at least "
      f"{MIN_SEED_WIN_FRACTION:.0%} of seeds AND wins at least "
      f"{MIN_FOLD_WIN_FRACTION:.0%} of all individual folds.")

# ----------------------------------------- A. repeated cross-validation
print(f"\n=== A. Repeated cross-validation ({args.seeds} seeds x 5 folds) ===")
models = {
    "Random forest (baseline)": make_model,
    "Hist. gradient boosting": make_hist_gradient_boosting,
    "Logistic regression (scaled)": make_logistic_regression,
}
fold_scores = {}   # name -> array (seeds, folds)
for name, factory in models.items():
    print(f"Running {name} ...")
    per_seed = []
    for seed in range(args.seeds):
        _, fold_f1, _ = cross_validate(factory, X, y, groups, seed=seed)
        per_seed.append(fold_f1)
    fold_scores[name] = np.array(per_seed)

baseline_name = "Random forest (baseline)"
base = fold_scores[baseline_name]
rows = []
for name, scores in fold_scores.items():
    row = {
        "model": name,
        "macro_f1_mean": float(scores.mean()),
        "spread_across_seeds": float(scores.mean(axis=1).std()),
    }
    if name != baseline_name:
        paired = paired_comparison(base.ravel(), scores.ravel())
        seed_wins = seed_win_fraction(base.mean(axis=1), scores.mean(axis=1))
        row["mean_diff_vs_baseline"] = paired["mean_diff"]
        row["fold_win_fraction"] = paired["fold_win_fraction"]
        row["seed_win_fraction"] = seed_wins
        row["replaces_baseline"] = passes_decision_rule(
            seed_wins, paired["fold_win_fraction"])
    rows.append(row)
selection = pd.DataFrame(rows)
selection.to_csv(REPORT_DIR / "model_selection.csv", index=False)
print()
print(selection.round(3).to_string(index=False))
print("\nThe spread across seeds shows how much the score depends on which "
      "blocks land in which fold.")
print("Even a clear winner was chosen using these same folds, so its score "
      "is somewhat optimistic.")

# ------------------------------------------------ B. sensor-group ablation
if args.skip_ablation:
    raise SystemExit(0)

print(f"\n=== B. Sensor-group ablation "
      f"({args.ablation_seeds} seeds x 5 folds) ===")
print("Question: does removing some sensors reduce the accuracy drop when "
      "the cooler is near total failure?")
ablation_models = {
    "Random forest": make_model,
    "Hist. gradient boosting": make_hist_gradient_boosting,
}
rows = []
for group_name, sensors in SENSOR_GROUPS.items():
    columns = columns_for_sensors(X.columns, sensors)
    X_group = X[columns]
    for model_name, factory in ablation_models.items():
        print(f"Running {model_name} / {group_name} ({len(columns)} features)")
        f1_all, acc_failing, acc_other = [], [], []
        for seed in range(args.ablation_seeds):
            oof, fold_f1, _ = cross_validate(
                factory, X_group, y, groups, seed=seed)
            f1_all.extend(fold_f1)
            acc_failing.append(subgroup_accuracy(y, oof, cooler_failing))
            acc_other.append(subgroup_accuracy(y, oof, ~cooler_failing))
        rows.append({
            "model": model_name,
            "sensor_group": group_name,
            "features": len(columns),
            "macro_f1_mean": float(np.mean(f1_all)),
            "accuracy_cooler_3": float(np.mean(acc_failing)),
            "accuracy_cooler_other": float(np.mean(acc_other)),
            "gap": float(np.mean(acc_other) - np.mean(acc_failing)),
        })
ablation = pd.DataFrame(rows)
ablation.to_csv(REPORT_DIR / "sensor_ablation.csv", index=False)
print()
print(ablation.round(3).to_string(index=False))
print("\nA smaller 'gap' means the valve estimate depends less on the "
      "cooler's condition. Removing sensors that lower macro F1 a lot is "
      "not a good trade.")
print("Saved: model_selection.csv and sensor_ablation.csv in reports/")