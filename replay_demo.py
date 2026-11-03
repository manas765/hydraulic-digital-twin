import json

from src.hydraulic_twin.twin_state import DigitalTwin

twin = DigitalTwin(desired_flow_lpm=8.0)
targets = twin.targets


def first_cycle(valve_label, stable_flag):
    match = targets[
        (targets["valve_condition"] == valve_label)
        & (targets["stable_flag"] == stable_flag)
    ]
    return int(match.index[0])


# (cycle, inject a simulated sensor fault?, description)
plan = [
    (first_cycle(100, 0), False, "stable, valve 100"),
    (first_cycle(90, 0), False, "stable, valve 90"),
    (first_cycle(80, 0), False, "stable, valve 80"),
    (first_cycle(73, 0), False, "stable, valve 73"),
    (first_cycle(100, 1), False, "UNSTABLE flag, valve 100"),
    (first_cycle(100, 0), True, "simulated sensor fault"),
]

print("Dataset replay (historical cycles, not live data).")
print("dataset label / stable flag = UCI data; predicted = ML estimate;")
print("mode, opening, flow, pressure = SIMULATED.\n")
header = (f"{'cycle':>5} {'label':>5} {'stbl':>4} {'pred':>5} {'conf':>5} "
          f"{'mode':<22} {'open%':>6} {'flow':>6} {'bar':>6} {'alerts':>6}")
print(header)

states = []
for cycle_id, fault, description in plan:
    state = twin.advance_to_cycle(cycle_id, simulate_sensor_fault=fault)
    states.append(state)

    pred = state["predicted_valve_condition"]["value"]
    probs = state["valve_condition_probabilities"]["value"]
    conf = f"{max(probs.values()):.2f}" if probs else "  -"
    print(f"{cycle_id:5d} "
          f"{state['dataset_labels']['valve_condition']['value']:5d} "
          f"{state['dataset_labels']['stable_flag']['value']:4d} "
          f"{str(pred) if pred is not None else '-':>5} {conf:>5} "
          f"{state['control_mode']['value']:<22} "
          f"{state['simulated_valve_opening_percent']['value']:6.1f} "
          f"{state['simulated_valve_flow_lpm']['value']:6.2f} "
          f"{state['simulated_pressure_bar']['value']:6.1f} "
          f"{len(state['alerts']):6d}   <- {description}")

print("\nAlerts raised:")
for state in states:
    for alert in state["alerts"]:
        print(f"  cycle {state['cycle_id']:4d} [{alert['level']}] "
              f"({alert['source']}) {alert['message']}")

example = states[2]  # the valve-80 cycle
print("\nOne full tagged state (valve-80 cycle), observed sensors shortened:")
shown = {k: v for k, v in example.items() if k != "observed_sensors"}
shown["observed_sensors"] = {
    name: example["observed_sensors"][name] for name in ("PS1", "FS1", "TS1")
}
print(json.dumps(shown, indent=2))