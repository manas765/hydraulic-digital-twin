import math
from dataclasses import dataclass


@dataclass
class ProportionalValve:
    """Normalized virtual proportional valve. SIMULATION ONLY.

    All parameters below are ASSUMED illustration values. They are not taken
    from the UCI dataset and are not specifications of any real valve.
    0% = commanded closed, 100% = commanded fully open.
    """

    max_flow_lpm: float = 20.0                  # ASSUMED
    opening_percent: float = 0.0                # SIMULATED state
    response_time_constant_s: float = 0.4       # ASSUMED first-order lag
    reference_pressure_drop_bar: float = 100.0  # ASSUMED normalization

    def command(self, requested_percent: float, dt_seconds: float = 0.1) -> float:
        """Move the simulated opening toward a bounded request."""
        if dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive")

        requested = max(0.0, min(100.0, requested_percent))
        alpha = 1.0 - math.exp(-dt_seconds / self.response_time_constant_s)
        self.opening_percent += alpha * (requested - self.opening_percent)
        self.opening_percent = max(0.0, min(100.0, self.opening_percent))
        return self.opening_percent

    def estimate_flow(self, pressure_drop_bar: float) -> float:
        """Illustrative normalized flow estimate, not calibrated hardware data."""
        if pressure_drop_bar <= 0:
            return 0.0

        opening_ratio = self.opening_percent / 100.0
        pressure_factor = min(
            1.0,
            math.sqrt(pressure_drop_bar / self.reference_pressure_drop_bar),
        )
        return self.max_flow_lpm * opening_ratio * pressure_factor