import numpy as np
import pandas as pd

from src.hydraulic_twin.ml_experiments import (
    class_recalls,
    cross_validate,
    make_logistic_regression,
    subgroup_report,
)


def test_class_recalls_computed_per_class():
    recalls = class_recalls(
        [100, 100, 90, 90, 80], [100, 90, 90, 90, 80]
    )
    assert recalls["recall_100"] == 0.5
    assert recalls["recall_90"] == 1.0
    assert recalls["recall_80"] == 1.0
    assert np.isnan(recalls["recall_73"])


def test_subgroup_report_splits_by_component_label():
    results = pd.DataFrame({
        "true_valve_condition": [100, 100, 90, 90],
        "oof_predicted_valve_condition": [100, 90, 90, 90],
        "pump_leakage": [0, 0, 1, 1],
    })
    report = subgroup_report(results, "pump_leakage")
    row0 = report[report["label"] == 0].iloc[0]
    row1 = report[report["label"] == 1].iloc[0]
    assert row0["cycles"] == 2 and row0["accuracy"] == 0.5
    assert row1["accuracy"] == 1.0


def test_cross_validate_returns_one_prediction_per_cycle():
    rng = np.random.default_rng(0)
    n = 400
    y = pd.Series(np.repeat([73, 80, 90, 100], n // 4))
    X = pd.DataFrame({
        "a": y.to_numpy() + rng.normal(size=n),
        "b": rng.normal(size=n),
    })
    groups = np.arange(n) // 20
    oof, fold_f1, fold_bal = cross_validate(
        make_logistic_regression, X, y, groups, n_splits=4
    )
    assert len(oof) == n
    assert set(np.unique(oof)) <= {73, 80, 90, 100}
    assert len(fold_f1) == 4 and len(fold_bal) == 4