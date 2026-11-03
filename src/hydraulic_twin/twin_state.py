import numpy as np
import pandas as pd

from .config import (
    PROCESSED_DATA_DIR,
    PROFILE_COLUMNS,
    SENSOR_RATES,
    SENSOR_UNITS,
)
from .controller import control_decision
from .plant_model import HydraulicPlant
from .predict import predict_valve_health
from .proportional_valve import ProportionalValve

DATA_SOURCE = "UCI_DATASET_REPLAY"

# Where each value comes from. Never mix these up in a display or report.
SRC_DATASET = "dataset"                  # measured by the UCI rig (cycle summary)
SRC_ML = "ml_prediction"                 # estimated by our classifier
SRC_SIM = "simulation"                   # produced by our virtual valve/plant
SRC_CONTROL = "controller_simulation"    # decided by our demo controller
SRC_USER = "user_setting"                # chosen by the person running the demo

LOW_CONFIDENCE_THRESHOLD = 0.60          # ASSUMED alert threshold
FLOW_SHORTFALL_ALERT_LPM = 1.0           # ASSUMED alert threshold

REPLAY_NOTE = (
    "Replayed historical dataset cycles, not live data. The final model was "
    "trained on every stable cycle, so a stable replayed cycle is training "
    "data: agreement with its label here checks the plumbing, NOT accuracy. "
    "Use the cross-validation results for accuracy."
)


def tag(source: str, value, unit: str = None, **extra) -> dict:
    """Wrap a value with its origin so it can never be mistaken."""
    item = {"source": source, "value": value}
    if unit is not None:
        item["unit"] = unit
    item.update(extra)
    return item


class DigitalTwin:
    """Offline replay twin: observed dataset values + ML estimate +
    simulated valve/plant, kept separate and labelled by source."""

    def __init__(
        self,
        desired_flow_lpm: float = 8.0,
        sim_seconds_per_cycle: float = 60.0,
        dt_seconds: float = 0.1,
    ):
        if dt_seconds <= 0 or sim_seconds_per_cycle < dt_seconds:
            raise ValueError("Need 0 < dt_seconds <= sim_seconds_per_cycle")

        self.features = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
        self.targets = pd.read_csv(PROCESSED_DATA_DIR / "targets.csv")
        if len(self.features) != len(self.targets):
            raise ValueError("Features and targets have different row counts")

        self.desired_flow_lpm = desired_flow_lpm
        self.sim_seconds_per_cycle = sim_seconds_per_cycle
        self.dt_seconds = dt_seconds
        self.reset_simulation()

    @property
    def n_cycles(self) -> int:
        return len(self.features)

    def reset_simulation(self):
        self.valve = ProportionalValve()
        self.plant = HydraulicPlant()
        self.simulation_time_s = 0.0

    def advance_to_cycle(
        self, cycle_id: int, simulate_sensor_fault: bool = False
    ) -> dict:
        """Replay one dataset cycle and advance the simulation by one cycle
        length. Simulation state carries over from the previous call."""
        if not 0 <= cycle_id < self.n_cycles:
            raise IndexError(f"cycle_id must be 0..{self.n_cycles - 1}")

        row = self.features.iloc[[cycle_id]]
        labels = self.targets.iloc[cycle_id]
        values = row.drop(columns=["cycle_id"]).to_numpy(dtype=float)

        data_finite = bool(np.isfinite(values).all())
        sensor_valid = data_finite and not simulate_sensor_fault

        prediction = predict_valve_health(row) if sensor_valid else None
        condition = prediction["predicted_condition"] if prediction else -1

        # Advance the simulation (valve and plant state persist across cycles)
        steps = int(round(self.sim_seconds_per_cycle / self.dt_seconds))
        decision, valve_flow = None, 0.0
        for _ in range(steps):
            decision = control_decision(
                desired_flow_lpm=self.desired_flow_lpm,
                pressure_bar=self.plant.pressure_bar,
                sensor_valid=sensor_valid,
                predicted_valve_condition=condition,
                current_opening_percent=self.valve.opening_percent,
            )
            self.valve.command(
                decision["requested_opening_percent"], self.dt_seconds
            )
            valve_flow = self.valve.estimate_flow(self.plant.pressure_bar)
            self.plant.step(valve_flow, self.dt_seconds)
        self.simulation_time_s += steps * self.dt_seconds

        alerts = self._build_alerts(
            labels, prediction, decision, valve_flow,
            simulate_sensor_fault, data_finite,
        )

        observed = {
            sensor: tag(
                SRC_DATASET,
                float(row[f"{sensor}_mean"].iloc[0]),
                SENSOR_UNITS[sensor],
                aggregation="cycle mean",
            )
            for sensor in SENSOR_RATES
        }

        return {
            "cycle_id": int(cycle_id),
            "data_source": DATA_SOURCE,
            "simulation_time_s": round(self.simulation_time_s, 3),
            "observed_sensors": observed,
            "dataset_labels": {
                name: tag(SRC_DATASET, int(labels[name]))
                for name in PROFILE_COLUMNS
            },
            "predicted_valve_condition": tag(
                SRC_ML, condition if prediction else None
            ),
            "valve_condition_probabilities": tag(
                SRC_ML, prediction["class_probabilities"] if prediction else {}
            ),
            "sensor_valid": tag(SRC_SIM if simulate_sensor_fault
                                else SRC_DATASET, sensor_valid),
            "desired_flow_lpm": tag(SRC_USER, self.desired_flow_lpm, "L/min"),
            "control_mode": tag(
                SRC_CONTROL, decision["mode"], reason=decision["reason"]
            ),
            "requested_valve_opening_percent": tag(
                SRC_CONTROL, round(decision["requested_opening_percent"], 3),
                "%",
            ),
            "simulated_valve_opening_percent": tag(
                SRC_SIM, round(self.valve.opening_percent, 3), "%"
            ),
            "simulated_valve_flow_lpm": tag(
                SRC_SIM, round(valve_flow, 3), "L/min"
            ),
            "simulated_pressure_bar": tag(
                SRC_SIM, round(self.plant.pressure_bar, 3), "bar"
            ),
            "simulated_relief_flow_lpm": tag(
                SRC_SIM, round(self.plant.relief_flow_lpm, 3), "L/min"
            ),
            "alerts": alerts,
            "replay_info": {
                "model_trained_on_this_cycle": int(labels["stable_flag"]) == 0,
                "note": REPLAY_NOTE,
            },
        }

    def _build_alerts(
        self, labels, prediction, decision, valve_flow,
        simulate_sensor_fault, data_finite,
    ) -> list:
        alerts = []

        def add(level, source, message):
            alerts.append(
                {"level": level, "source": source, "message": message}
            )

        if simulate_sensor_fault:
            add("WARNING", SRC_SIM,
                "Simulated sensor fault injected; ML prediction skipped")
        elif not data_finite:
            add("WARNING", SRC_DATASET,
                "Non-finite feature values; ML prediction skipped")

        if int(labels["stable_flag"]) == 1:
            add("WARNING", SRC_DATASET,
                "Dataset flags this cycle as possibly not yet stable. The "
                "model was trained on stable cycles only, so this prediction "
                "is outside its training range")

        if prediction:
            condition = prediction["predicted_condition"]
            if condition == 73:
                add("CRITICAL", SRC_ML,
                    "Valve predicted close to total failure "
                    "(ML estimate; needs review)")
            elif condition == 80:
                add("WARNING", SRC_ML,
                    "Valve predicted severe switching lag (ML estimate)")
            elif condition == 90:
                add("INFO", SRC_ML,
                    "Valve predicted small switching lag (ML estimate)")

            top = max(prediction["class_probabilities"].values())
            if top < LOW_CONFIDENCE_THRESHOLD:
                add("INFO", SRC_ML,
                    f"Low model confidence (top class probability {top:.2f})")

        if decision["mode"] != "NORMAL_SIMULATION":
            add("WARNING", SRC_CONTROL,
                f"Controller mode {decision['mode']}: {decision['reason']}")

        shortfall = self.desired_flow_lpm - valve_flow
        if shortfall > FLOW_SHORTFALL_ALERT_LPM:
            add("INFO", SRC_SIM,
                f"Simulated flow is {shortfall:.1f} L/min below demand")

        return alerts