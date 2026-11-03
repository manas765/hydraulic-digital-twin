from dataclasses import dataclass

VALID_CONDITIONS = {73, 80, 90, 100}
RESTRICTED_MAX_OPENING_PERCENT = 25.0  # ASSUMED


@dataclass
class ControlLimits:
    """All values are ASSUMED simulation settings, not rig specifications."""

    min_pressure_bar: float = 5.0
    max_pressure_bar: float = 140.0
    max_flow_lpm: float = 18.0          # largest demand accepted
    valve_rated_flow_lpm: float = 20.0  # nominal valve size used to convert
                                        # demand to an opening request


def control_decision(
    desired_flow_lpm: float,
    pressure_bar: float,
    sensor_valid: bool,
    predicted_valve_condition: int,
    current_opening_percent: float,
    limits: ControlLimits = ControlLimits(),
) -> dict:
    """Software-only demonstration controller. Commands no real equipment.

    pressure_bar is the SIMULATED plant pressure, not a dataset reading.
    current_opening_percent is accepted for later rate limiting / PID use;
    this version does not use it.
    """
    if not sensor_valid:
        return {
            "mode": "INHIBIT",
            "requested_opening_percent": 0.0,
            "reason": "Sensor data invalid",
        }

    if not (limits.min_pressure_bar <= pressure_bar <= limits.max_pressure_bar):
        return {
            "mode": "SAFE_SIMULATION_STATE",
            "requested_opening_percent": 0.0,
            "reason": "Pressure outside configured simulation limits",
        }

    if desired_flow_lpm < 0 or desired_flow_lpm > limits.max_flow_lpm:
        return {
            "mode": "INHIBIT",
            "requested_opening_percent": 0.0,
            "reason": "Requested flow outside simulation limits",
        }

    if predicted_valve_condition not in VALID_CONDITIONS:
        return {
            "mode": "INHIBIT",
            "requested_opening_percent": 0.0,
            "reason": "Unknown predicted valve condition",
        }

    if predicted_valve_condition == 73:
        return {
            "mode": "FAULT_REVIEW",
            "requested_opening_percent": 0.0,
            "reason": "Valve predicted close to total failure",
        }

    requested = 100.0 * desired_flow_lpm / limits.valve_rated_flow_lpm
    requested = max(0.0, min(100.0, requested))

    if predicted_valve_condition == 80:
        return {
            "mode": "RESTRICTED_SIMULATION",
            "requested_opening_percent": min(
                requested, RESTRICTED_MAX_OPENING_PERCENT
            ),
            "reason": "Severe predicted valve lag; opening capped (demo mode)",
        }

    return {
        "mode": "NORMAL_SIMULATION",
        "requested_opening_percent": requested,
        "reason": "Demand converted to proportional opening request",
    }