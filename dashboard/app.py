from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hydraulic_twin.config import (  # noqa: E402
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    PROFILE_COLUMNS,
    SENSOR_RATES,
    SENSOR_UNITS,
)
from src.hydraulic_twin.controller import ControlLimits  # noqa: E402
from src.hydraulic_twin.dashboard_data import (  # noqa: E402
    COMPONENT_TITLES,
    FLOW_COLUMNS,
    OPENING_COLUMNS,
    PRESSURE_COLUMNS,
    SOURCE_LABELS,
    confusion_frame,
    describe_label,
    example_cycles,
    history_frame,
    history_window,
    load_cv_results,
    load_oof_predictions,
    report_frame,
    sensor_history,
    sensor_table,
    trace_frame,
)
from src.hydraulic_twin.twin_state import DigitalTwin  # noqa: E402

st.set_page_config(page_title="Hydraulic Digital Twin", layout="wide")

REQUIRED_FILES = [
    PROCESSED_DATA_DIR / "features.csv",
    PROCESSED_DATA_DIR / "targets.csv",
    MODEL_DIR / "valve_health_pipeline.joblib",
    MODEL_DIR / "valve_health_features.json",
]
missing = [p.name for p in REQUIRED_FILES if not p.exists()]
if missing:
    st.error(
        "Missing files: " + ", ".join(missing) + ". Run the preprocessing "
        "and training steps from the README first."
    )
    st.stop()


@st.cache_resource
def get_twin():
    return DigitalTwin()


@st.cache_data(show_spinner=False)
def run_twin(cycle_id: int, desired_flow_lpm: float, sensor_fault: bool):
    """Each view simulates the selected cycle from the same starting
    condition (valve closed, plant at its initial pressure), so what you
    see does not depend on how you navigated to this cycle."""
    twin = get_twin()
    twin.desired_flow_lpm = desired_flow_lpm
    twin.reset_simulation()
    state = twin.advance_to_cycle(
        cycle_id, simulate_sensor_fault=sensor_fault
    )
    return state, twin.last_trace


twin = get_twin()
n_cycles = twin.n_cycles
examples = example_cycles(twin.targets)

if "cycle" not in st.session_state:
    st.session_state.cycle = next(iter(examples.values()), 0)


def go_to(cycle_id):
    st.session_state.cycle = int(cycle_id) % n_cycles


def step(delta):
    st.session_state.cycle = (st.session_state.cycle + delta) % n_cycles


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Replay controls")
    st.caption("DATASET REPLAY: historical cycles, not live data.")
    st.slider("Dataset cycle", min_value=0, max_value=n_cycles - 1,
              key="cycle")
    col_prev, col_next = st.columns(2)
    col_prev.button("Previous", on_click=step, args=(-1,))
    col_next.button("Next", on_click=step, args=(1,))

    st.subheader("Quick jump")
    for name, cid in examples.items():
        st.button(f"{name}: cycle {cid}", on_click=go_to, args=(cid,),
                  key=f"jump_{name}")

    st.header("Simulation inputs")
    desired_flow = st.slider(
        "Desired flow [L/min] (USER-SELECTED DEMAND)",
        min_value=0.0,
        max_value=float(ControlLimits().max_flow_lpm),
        value=8.0,
        step=0.5,
    )
    sensor_fault = st.checkbox("Inject simulated sensor fault")
    st.caption("The fault is simulated. It is not in the dataset.")

cycle = int(st.session_state.cycle)
state, trace = run_twin(cycle, float(desired_flow), bool(sensor_fault))

predicted = state["predicted_valve_condition"]["value"]
probabilities = state["valve_condition_probabilities"]["value"]
confidence = max(probabilities.values()) if probabilities else None
mode = state["control_mode"]["value"]
reason = state["control_mode"]["reason"]
opening = state["simulated_valve_opening_percent"]["value"]
valve_flow = state["simulated_valve_flow_lpm"]["value"]
pressure = state["simulated_pressure_bar"]["value"]
labels = {k: v["value"] for k, v in state["dataset_labels"].items()}

# ------------------------------------------------------------------ header
st.title("Hydraulic System - Digital Twin")
st.caption(
    "DATASET REPLAY of historical UCI cycles | ML PREDICTION | SIMULATED "
    "valve and plant. Not live data. Commands no real equipment."
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Replay cycle", f"{cycle} / {n_cycles - 1}")
m2.metric("ML PREDICTION: valve condition",
          f"{predicted}%" if predicted is not None else "n/a")
if confidence is not None:
    m2.caption(f"Top class probability {confidence:.0%}")
m3.metric("SIMULATED VALUE: valve opening", f"{opening:.1f}%")
m4.metric("SIMULATED VALUE: valve flow", f"{valve_flow:.2f} L/min")
m5.metric("SIMULATED VALUE: pressure", f"{pressure:.1f} bar")

st.markdown(
    f"**Control mode (SIMULATED, controller):** `{mode}` - {reason}"
)


def show_alert(alert):
    source = SOURCE_LABELS.get(alert["source"], alert["source"])
    text = f"**{alert['level']}** | {source} | {alert['message']}"
    if alert["level"] == "CRITICAL":
        st.error(text)
    elif alert["level"] == "WARNING":
        st.warning(text)
    else:
        st.info(text)


for alert in state["alerts"]:
    show_alert(alert)

(tab_overview, tab_sensors, tab_health, tab_twin, tab_faults,
 tab_experiments) = st.tabs([
    "Overview", "Sensors", "Valve health", "Twin and control",
    "Faults", "Experiments",
])

# ---------------------------------------------------------------- overview
with tab_overview:
    st.subheader("Everything shown for this cycle, with its source")
    rows = [
        ("Dataset valve label (offline evaluation only)",
         f"{labels['valve_condition']}%", SOURCE_LABELS["dataset"]),
        ("Stable flag (0 = stable)", labels["stable_flag"],
         SOURCE_LABELS["dataset"]),
        ("Predicted valve condition",
         f"{predicted}%" if predicted is not None else "n/a",
         SOURCE_LABELS["ml_prediction"]),
        ("Desired flow", f"{desired_flow:.1f} L/min",
         SOURCE_LABELS["user_setting"]),
        ("Control mode", mode, SOURCE_LABELS["controller_simulation"]),
        ("Requested valve opening",
         f"{state['requested_valve_opening_percent']['value']:.1f}%",
         SOURCE_LABELS["controller_simulation"]),
        ("Simulated valve opening", f"{opening:.1f}%",
         SOURCE_LABELS["simulation"]),
        ("Simulated valve flow", f"{valve_flow:.2f} L/min",
         SOURCE_LABELS["simulation"]),
        ("Simulated pressure", f"{pressure:.1f} bar",
         SOURCE_LABELS["simulation"]),
        ("Simulated relief-valve flow",
         f"{state['simulated_relief_flow_lpm']['value']:.2f} L/min",
         SOURCE_LABELS["simulation"]),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["Quantity", "Value", "Source"]),
        hide_index=True,
    )
    if state["replay_info"]["model_trained_on_this_cycle"]:
        st.info(state["replay_info"]["note"])
    st.caption(
        "Dataset: Helwig, Pignanelli and Schuetze (2015), Condition "
        "monitoring of hydraulic systems, UCI Machine Learning Repository, "
        "DOI 10.24432/C5CW21, CC BY 4.0."
    )

# ----------------------------------------------------------------- sensors
with tab_sensors:
    st.subheader("Observed sensors (this cycle)")
    st.caption(
        "DATASET MEASUREMENT: each value is the mean of the sensor over "
        "this 60-second cycle, not a live reading."
    )
    st.dataframe(sensor_table(state), hide_index=True)

    st.subheader("History")
    chosen_sensor = st.selectbox("Sensor", list(SENSOR_RATES), key="hist_s")
    half_width = st.slider("Cycles shown either side", 20, 300, 100,
                           key="hist_w")
    start, stop = history_window(n_cycles, cycle, half_width)
    st.line_chart(sensor_history(
        twin.features, chosen_sensor, SENSOR_UNITS[chosen_sensor],
        start, stop,
    ))
    st.caption(
        f"DATASET MEASUREMENT: cycles {start} to {stop - 1}. The selected "
        f"cycle is {cycle}."
    )

# ------------------------------------------------------------ valve health
with tab_health:
    st.subheader("ML PREDICTION: valve condition")
    if predicted is None:
        st.warning("No prediction: sensor data invalid (simulated fault).")
    else:
        st.write(f"**Predicted:** {predicted}% - "
                 f"{describe_label('valve_condition', predicted)} "
                 "(an estimate, not a measurement)")
        for label in ("100", "90", "80", "73"):
            p = float(min(1.0, max(0.0, probabilities.get(label, 0.0))))
            meaning = describe_label("valve_condition", int(label))
            st.progress(p, text=f"{label}% ({meaning}): {p:.1%}")

        actual = labels["valve_condition"]
        if actual == predicted:
            st.success(f"Matches the dataset label ({actual}%).")
        else:
            st.warning(f"Differs from the dataset label ({actual}%).")
        st.caption(
            "The dataset label is shown for offline evaluation only. A real "
            "deployment would not have it. If this cycle was in the model's "
            "training data, agreement says nothing about accuracy."
        )

    st.subheader("History: dataset label vs out-of-fold ML prediction")
    oof = load_oof_predictions("stable_only")
    if oof is None:
        st.info("Run the training step to create out-of-fold predictions.")
    else:
        start, stop = history_window(n_cycles, cycle, 100)
        st.line_chart(history_frame(twin.targets, oof, start, stop))
        st.caption(
            "Out-of-fold: each prediction comes from a model that never saw "
            "that cycle's block of 50 during training. Gaps are cycles the "
            "stable-only baseline does not cover (unstable cycles)."
        )

# ---------------------------------------------------------- twin & control
with tab_twin:
    st.subheader("Control simulation")
    c1, c2, c3 = st.columns(3)
    c1.metric("Control mode (SIMULATED)", mode)
    c2.metric(
        "Requested opening (SIMULATED)",
        f"{state['requested_valve_opening_percent']['value']:.1f}%",
    )
    c3.metric("Desired flow (USER-SELECTED DEMAND)",
              f"{desired_flow:.1f} L/min")
    st.caption(reason)

    frame = trace_frame(trace, desired_flow)
    st.subheader("Simulated response over this cycle (60 s)")
    st.caption(
        "SIMULATED VALUE. Starts from valve closed and the plant's initial "
        "pressure. Assumed parameters are listed in the README."
    )
    st.line_chart(frame[OPENING_COLUMNS])
    st.line_chart(frame[FLOW_COLUMNS])
    st.line_chart(frame[PRESSURE_COLUMNS])

    st.subheader("Observed vs simulated")
    comparison = pd.DataFrame([
        {"Quantity": "Pressure (PS1 cycle mean)",
         "Value": state["observed_sensors"]["PS1"]["value"],
         "Unit": "bar", "Source": SOURCE_LABELS["dataset"]},
        {"Quantity": "Pressure (simulated plant)",
         "Value": pressure, "Unit": "bar",
         "Source": SOURCE_LABELS["simulation"]},
        {"Quantity": "Flow (FS1 cycle mean)",
         "Value": state["observed_sensors"]["FS1"]["value"],
         "Unit": "L/min", "Source": SOURCE_LABELS["dataset"]},
        {"Quantity": "Flow (simulated valve)",
         "Value": valve_flow, "Unit": "L/min",
         "Source": SOURCE_LABELS["simulation"]},
    ])
    st.dataframe(comparison, hide_index=True)
    st.caption(
        "These are not expected to match. The simulation uses assumed "
        "parameters and is not calibrated to the rig."
    )

# ------------------------------------------------------------------ faults
with tab_faults:
    st.subheader("Component labels for this cycle")
    component_rows = [
        {
            "Component": COMPONENT_TITLES[name],
            "Dataset label": labels[name],
            "Meaning": describe_label(name, labels[name]),
            "Source": SOURCE_LABELS["dataset"],
        }
        for name in PROFILE_COLUMNS
        if name in COMPONENT_TITLES
    ]
    st.dataframe(pd.DataFrame(component_rows), hide_index=True)
    st.caption(
        "Dataset labels are shown for offline evaluation. A real "
        "deployment would not have them."
    )

    st.subheader("Alerts for this cycle")
    if state["alerts"]:
        st.dataframe(pd.DataFrame([
            {"Level": a["level"],
             "Source": SOURCE_LABELS.get(a["source"], a["source"]),
             "Message": a["message"]}
            for a in state["alerts"]
        ]), hide_index=True)
    else:
        st.success("No alerts for this cycle.")

# ------------------------------------------------------------- experiments
with tab_experiments:
    st.subheader("Cross-validation results")
    st.caption(
        "5-fold StratifiedGroupKFold; blocks of 50 consecutive cycles stay "
        "together. These are the honest accuracy numbers, not the replay."
    )

    shown_any = False
    for run_name, title in (
        ("stable_only", "Baseline: stable cycles only"),
        ("all_cycles", "Experiment: all cycles (stable and unstable)"),
    ):
        results = load_cv_results(run_name)
        if results is None:
            continue
        shown_any = True
        st.markdown(f"### {title}")
        r1, r2, r3 = st.columns(3)
        r1.metric("Macro F1",
                  f"{results['macro_f1_mean']:.3f} +/- "
                  f"{results['macro_f1_std']:.3f}")
        r2.metric("Balanced accuracy",
                  f"{results['balanced_accuracy_mean']:.3f} +/- "
                  f"{results['balanced_accuracy_std']:.3f}")
        r3.metric("Cycles", results["n_cycles"])

        st.write("Per-fold scores")
        st.dataframe(pd.DataFrame({
            "Fold": list(range(1, len(results["fold_macro_f1"]) + 1)),
            "Macro F1": results["fold_macro_f1"],
            "Balanced accuracy": results["fold_balanced_accuracy"],
        }), hide_index=True)

        st.write("Confusion matrix (rows = true, columns = predicted)")
        st.dataframe(confusion_frame(results))
        st.write("Per-class results")
        st.dataframe(report_frame(results), hide_index=True)

    if not shown_any:
        st.info("No saved results yet. Run the training step.")
    else:
        st.warning(
            "The two runs are not comparable: all 756 unstable cycles are "
            "valve class 100, which makes the all-cycles test set easier "
            "and lets a model use a warm-up shortcut."
        )