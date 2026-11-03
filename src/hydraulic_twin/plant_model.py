from dataclasses import dataclass


@dataclass
class HydraulicPlant:
    """Simplified normalized plant. SIMULATION ONLY.

    All parameters are ASSUMED demonstration values, not a calibrated model
    and not data from the UCI rig.

    A simple SIMULATED relief valve is included: above relief_setting_bar,
    surplus flow leaves through it in proportion to the pressure excess.
    This is a modelling assumption for the demo. It is NOT a substitute for
    real, independent pressure protection designed by qualified engineers.
    """

    pressure_bar: float = 50.0                    # SIMULATED state
    supply_pressure_bar: float = 150.0            # ASSUMED hard cap
    pump_flow_lpm: float = 12.0                   # ASSUMED
    leak_flow_lpm: float = 0.2                    # ASSUMED
    pressure_gain_bar_per_lpm_s: float = 0.8      # ASSUMED
    relief_setting_bar: float = 120.0             # ASSUMED
    relief_gain_lpm_per_bar: float = 2.0          # ASSUMED
    relief_flow_lpm: float = 0.0                  # SIMULATED state

    def step(self, valve_flow_lpm: float, dt_seconds: float = 0.1) -> float:
        if dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive")

        excess = max(0.0, self.pressure_bar - self.relief_setting_bar)
        self.relief_flow_lpm = self.relief_gain_lpm_per_bar * excess

        net_flow = (
            self.pump_flow_lpm
            - valve_flow_lpm
            - self.leak_flow_lpm
            - self.relief_flow_lpm
        )
        self.pressure_bar += (
            self.pressure_gain_bar_per_lpm_s * net_flow * dt_seconds
        )
        self.pressure_bar = max(
            0.0, min(self.supply_pressure_bar, self.pressure_bar)
        )
        return self.pressure_bar