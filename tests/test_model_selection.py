import numpy as np

from src.hydraulic_twin import model_selection as ms


def test_columns_for_sensors_matches_whole_sensor_names():
    columns = ["PS1_mean", "PS1_std", "PS10_mean", "TS1_mean", "EPS1_mean",
               "CE_mean", "CP_rms"]
    assert ms.columns_for_sensors(columns, ["PS1"]) == ["PS1_mean", "PS1_std"]
    assert ms.columns_for_sensors(columns, ["EPS1", "CE"]) == [
        "EPS1_mean", "CE_mean"]


def test_sensor_groups_remove_the_intended_sensors():
    no_temperature = ms.SENSOR_GROUPS["Without temperature (TS1-TS4)"]
    assert "TS1" not in no_temperature and "PS1" in no_temperature
    assert len(ms.SENSOR_GROUPS["All 17 sensors"]) == 17
    assert set(ms.SENSOR_GROUPS["Pressure, motor power, flow only"]) == set(
        ms.HYDRAULIC_ONLY)


def test_paired_comparison_counts_fold_wins():
    result = ms.paired_comparison([0.8, 0.8, 0.8, 0.8], [0.9, 0.9, 0.7, 0.9])
    assert result["n_folds"] == 4
    assert abs(result["mean_diff"] - 0.05) < 1e-9
    assert result["fold_win_fraction"] == 0.75


def test_decision_rule_needs_both_conditions():
    assert ms.passes_decision_rule(0.9, 0.8)
    assert not ms.passes_decision_rule(0.9, 0.5)
    assert not ms.passes_decision_rule(0.5, 0.9)


def test_seed_win_fraction():
    assert ms.seed_win_fraction([0.8, 0.8, 0.8, 0.8],
                                [0.9, 0.7, 0.9, 0.9]) == 0.75


def test_subgroup_accuracy_handles_empty_subgroup():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 0, 0])
    first_two = [True, True, False, False]
    assert ms.subgroup_accuracy(y_true, y_pred, first_two) == 1.0
    assert np.isnan(ms.subgroup_accuracy(y_true, y_pred, [False] * 4))