import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from .config import MODEL_DIR, PROCESSED_DATA_DIR, REPORT_DIR

VALVE_LABELS = [73, 80, 90, 100]
MODEL_PATH = MODEL_DIR / "valve_health_pipeline.joblib"
FEATURES_PATH = MODEL_DIR / "valve_health_features.json"


def make_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def save_cv_results(
    run_name, cycle_ids, y, oof_pred,
    fold_macro_f1, fold_bal_acc, block_size, n_splits,
):
    """Save metrics, confusion matrix and out-of-fold predictions so the
    dashboard can show them without retraining."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "run": run_name,
        "n_cycles": int(len(y)),
        "block_size": int(block_size),
        "n_splits": int(n_splits),
        "labels": VALVE_LABELS,
        "macro_f1_mean": float(np.mean(fold_macro_f1)),
        "macro_f1_std": float(np.std(fold_macro_f1)),
        "balanced_accuracy_mean": float(np.mean(fold_bal_acc)),
        "balanced_accuracy_std": float(np.std(fold_bal_acc)),
        "fold_macro_f1": [float(v) for v in fold_macro_f1],
        "fold_balanced_accuracy": [float(v) for v in fold_bal_acc],
        "confusion_matrix": confusion_matrix(
            y, oof_pred, labels=VALVE_LABELS
        ).tolist(),
        "classification_report": classification_report(
            y, oof_pred, labels=VALVE_LABELS,
            zero_division=0, output_dict=True,
        ),
    }
    with open(REPORT_DIR / f"valve_cv_{run_name}.json", "w",
              encoding="utf-8") as f:
        json.dump(
            results, f, indent=2,
            default=lambda o: o.item() if hasattr(o, "item") else str(o),
        )

    pd.DataFrame({
        "cycle_id": cycle_ids,
        "true_valve_condition": y.to_numpy(),
        "oof_predicted_valve_condition": oof_pred,
    }).to_csv(
        REPORT_DIR / f"valve_oof_predictions_{run_name}.csv", index=False
    )


def main(include_unstable: bool, block_size: int, n_splits: int):
    X_df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    y_df = pd.read_csv(PROCESSED_DATA_DIR / "targets.csv")

    if len(X_df) != len(y_df):
        raise ValueError("Features and targets have different row counts")
    if not (X_df["cycle_id"].to_numpy() == np.arange(len(X_df))).all():
        raise ValueError("cycle_id is not 0..N-1; rows may be misaligned")

    if include_unstable:
        mask = pd.Series(True, index=y_df.index)
        print("Using ALL cycles (stable and unstable).")
    else:
        mask = y_df["stable_flag"] == 0
        print("Using STABLE cycles only (stable_flag == 0).")

    X_df = X_df.loc[mask].reset_index(drop=True)
    y = y_df.loc[mask, "valve_condition"].reset_index(drop=True)

    cycle_ids = X_df.pop("cycle_id").to_numpy()
    X = X_df
    groups = cycle_ids // block_size  # consecutive cycles share a group

    print(f"Cycles used: {len(X)}   Features: {X.shape[1]}")
    print(f"Block size: {block_size}   Groups: {len(np.unique(groups))}")
    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=42
    )

    oof_pred = np.zeros(len(y), dtype=int)  # out-of-fold predictions
    fold_macro_f1 = []
    fold_bal_acc = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups=groups), start=1
    ):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if set(y_train.unique()) != set(VALVE_LABELS):
            raise ValueError(f"Fold {fold}: training set misses a class")

        counts = y_test.value_counts().reindex(VALVE_LABELS, fill_value=0)
        print(f"\nFold {fold}: test counts "
              + ", ".join(f"{k}:{v}" for k, v in counts.items()))
        if (counts == 0).any():
            print("  WARNING: a class is absent from this test fold; "
                  "its per-fold score is not meaningful.")

        model = make_model()
        model.fit(X.iloc[train_idx], y_train)
        pred = model.predict(X.iloc[test_idx])
        oof_pred[test_idx] = pred

        f1 = f1_score(y_test, pred, average="macro", zero_division=0)
        bal = balanced_accuracy_score(y_test, pred)
        fold_macro_f1.append(f1)
        fold_bal_acc.append(bal)
        print(f"  macro F1 = {f1:.3f}   balanced accuracy = {bal:.3f}")

    print("\n=== Cross-validation summary ===")
    print(f"Macro F1:          {np.mean(fold_macro_f1):.3f} "
          f"+/- {np.std(fold_macro_f1):.3f}")
    print(f"Balanced accuracy: {np.mean(fold_bal_acc):.3f} "
          f"+/- {np.std(fold_bal_acc):.3f}")

    print("\nClassification report (all out-of-fold predictions):")
    print(classification_report(
        y, oof_pred, labels=VALVE_LABELS, zero_division=0
    ))
    print("Confusion matrix (rows = true, columns = predicted; "
          "order 73, 80, 90, 100):")
    print(confusion_matrix(y, oof_pred, labels=VALVE_LABELS))

    run_name = "all_cycles" if include_unstable else "stable_only"
    save_cv_results(
        run_name, cycle_ids, y, oof_pred,
        fold_macro_f1, fold_bal_acc, block_size, n_splits,
    )
    print(f"\nSaved CV results: reports/valve_cv_{run_name}.json "
          f"and valve_oof_predictions_{run_name}.csv")

    if include_unstable:
        print("\nExperiment run: model NOT saved.")
        return

    final_model = make_model()
    final_model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    print("\nFinal model trained on all stable cycles.")
    print("Saved:", MODEL_PATH)
    print("Saved:", FEATURES_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-unstable", action="store_true")
    parser.add_argument("--block-size", type=int, default=50)
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()
    main(args.include_unstable, args.block_size, args.splits)