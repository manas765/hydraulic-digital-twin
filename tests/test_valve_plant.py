import pytest

from src.hydraulic_twin.plant_model import HydraulicPlant
from src.hydraulic_twin.proportional_valve import ProportionalValve


def test_command_above_100_clamps_to_100():
    valve = ProportionalValve()
    for _ in range(500):
        position = valve.command(150)
    assert 99.9 < position <= 100.0


def test_command_below_0_clamps_to_0():
    valve = ProportionalValve(opening_percent=50.0)
    for _ in range(500):
        position = valve.command(-20)
    assert 0.0 <= position < 0.1


def test_valve_responds_gradually():
    valve = ProportionalValve()
    first = valve.command(100)
    assert 0.0 < first < 100.0


def test_response_does_not_depend_on_step_size():
    coarse = ProportionalValve()
    fine = ProportionalValve()
    for _ in range(10):
        coarse.command(100, dt_seconds=0.1)
    for _ in range(100):
        fine.command(100, dt_seconds=0.01)
    assert abs(coarse.opening_percent - fine.opening_percent) < 0.01


def test_closed_valve_has_zero_flow():
    valve = ProportionalValve(opening_percent=0)
    assert valve.estimate_flow(100) == 0


def test_zero_pressure_drop_gives_zero_flow():
    valve = ProportionalValve(opening_percent=100)
    assert valve.estimate_flow(0) == 0


def test_more_opening_gives_more_flow():
    low = ProportionalValve(opening_percent=20)
    high = ProportionalValve(opening_percent=80)
    assert high.estimate_flow(50) > low.estimate_flow(50)


def test_nonpositive_time_step_is_rejected():
    with pytest.raises(ValueError):
        ProportionalValve().command(50, dt_seconds=0)
    with pytest.raises(ValueError):
        HydraulicPlant().step(5.0, dt_seconds=-1)


def test_pressure_rises_when_pump_exceeds_outflow():
    plant = HydraulicPlant(pressure_bar=50.0)
    assert plant.step(0.0, 0.1) > 50.0


def test_pressure_falls_when_outflow_exceeds_pump():
    plant = HydraulicPlant(pressure_bar=50.0)
    assert plant.step(20.0, 0.1) < 50.0


def test_pressure_cannot_go_below_zero():
    plant = HydraulicPlant()
    for _ in range(5000):
        pressure = plant.step(50.0, 0.1)
    assert pressure == 0.0


def test_relief_valve_limits_pressure_with_closed_valve():
    plant = HydraulicPlant()
    for _ in range(5000):
        pressure = plant.step(0.0, 0.1)
    assert plant.relief_setting_bar < pressure < plant.supply_pressure_bar
    assert plant.relief_flow_lpm > 0


def test_hard_cap_applies_when_relief_is_disabled():
    plant = HydraulicPlant(relief_gain_lpm_per_bar=0.0)
    for _ in range(5000):
        pressure = plant.step(0.0, 0.1)
    assert pressure == plant.supply_pressure_bar