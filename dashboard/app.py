from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hydraulic_twin.config import (  # noqa: E402
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
    assumption_rows,
    condition_tone,
    confusion_frame,
    describe_label,
    example_cycles,
    history_frame,
    history_window,
    load_cv_results,
    load_oof_predictions,
    load_report_csv,
    pretty_mode,
    report_frame,
    robustness_findings,
    sensor_history,
    sensor_table,
    trace_frame,
)
from src.hydraulic_twin.three_view import (  # noqa: E402
    build_scene_data,
    render_html,
)
from src.hydraulic_twin.twin_state import DigitalTwin  # noqa: E402
from src.hydraulic_twin.ui_components import (  # noqa: E402
    STYLE,
    alerts_html,
    card,
    card_grid,
    footer_html,
    header_html,
    section_html,
)

APP_VERSION = "1.0"

st.set_page_config(
    page_title="Hydraulic System Condition Monitoring",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(STYLE, unsafe_allow_html=True)


def section(title):
    st.markdown(section_html(title), unsafe_allow_html=True)


REQUIRED_FILES = [
    PROCESSED_DATA_DIR / "features.csv",
    PROCESSED_DATA_DIR / "targets.csv",
]
missing = [p.name for p in REQUIRED_FILES if not p.exists()]
if missing:
    st.error(
        "Missing files: " + ", ".join(missing) + ". Run the preprocessing "
        "and training steps described in the README first."
    )
    st.stop()


@st.cache_resource
def get_twin():
    return DigitalTwin()


@st.cache_data(show_spinner=False)
def run_twin(cycle_id: int, desired_flow_lpm: float, sensor_fault: bool):
    """Each view simulates the selected cycle from the same starting
    condition (valve closed, plant at its initial pressure), so what is
    shown does not depend on how the user navigated to this cycle."""
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
PLACEHOLDER = "Select an example..."

if "cycle" not in st.session_state:
    st.session_state.cycle = next(iter(examples.values()), 0)


def go_to(cycle_id):
    st.session_state.cycle = int(cycle_id) % n_cycles


def step(delta):
    st.session_state.cycle = (st.session_state.cycle + delta) % n_cycles


def load_example():
    choice = st.session_state.example
    if choice in examples:
        go_to(examples[choice])
    st.session_state.example = PLACEHOLDER


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("**Replay**")
    st.slider("Dataset cycle", min_value=0, max_value=n_cycles - 1,
              key="cycle")
    col_prev, col_next = st.columns(2)
    col_prev.button("Previous", on_click=step, args=(-1,))
    col_next.button("Next", on_click=step, args=(1,))
    st.selectbox("Example cycles", [PLACEHOLDER] + list(examples),
                 key="example", on_change=load_example)

    st.markdown("**Simulation inputs**")
    desired_flow = st.slider(
        "Desired flow [L/min]",
        min_value=0.0,
        max_value=float(ControlLimits().max_flow_lpm),
        value=8.0,
        step=0.5,
    )
    sensor_fault = st.checkbox("Simulate sensor fault")
    st.caption(
        "These are user settings. Dataset values cannot be edited. "
        "The sensor fault is simulated and is not in the dataset."
    )

cycle = int(st.session_state.cycle)
state, trace = run_twin(cycle, float(desired_flow), bool(sensor_fault))

predicted = state["predicted_valve_condition"]["value"]
probabilities = state["valve_condition_probabilities"]["value"]
confidence = max(probabilities.values()) if probabilities else None
mode = state["control_mode"]["value"]
reason = state["control_mode"]["reason"]
requested = state["requested_valve_opening_percent"]["value"]
opening = state["simulated_valve_opening_percent"]["value"]
valve_flow = state["simulated_valve_flow_lpm"]["value"]
pressure = state["simulated_pressure_bar"]["value"]
relief_flow = state["simulated_relief_flow_lpm"]["value"]
labels = {k: v["value"] for k, v in state["dataset_labels"].items()}
unstable = labels["stable_flag"] == 1


def show_html(html, height):
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:  # older Streamlit versions
        import streamlit.components.v1 as components
        components.html(html, height=height)


def pkg_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "n/a"


# ------------------------------------------------------------------ header
st.markdown(
    header_html(
        "Hydraulic System Condition Monitoring",
        "Valve health estimation and control simulation | "
        f"dataset cycle {cycle} of {n_cycles - 1}",
        "DATASET REPLAY | HISTORICAL DATA, NOT LIVE",
    ),
    unsafe_allow_html=True,
)

(tab_overview, tab_condition, tab_simulation, tab_sensors, tab_3d,
 tab_evaluation, tab_about) = st.tabs([
    "Overview", "Valve condition", "Simulation", "Sensors", "3D view",
    "Model evaluation", "About",
])

# ---------------------------------------------------------------- overview
with tab_overview:
    if predicted is None:
        condition_value, condition_sub = "n/a", "No estimate: data invalid"
    else:
        condition_value = f"{predicted} %"
        condition_sub = (
            f"{describe_label('valve_condition', predicted)} | "
            f"confidence {confidence:.0%}"
        )

    cards = [
        card("Valve condition", condition_value, "ml", condition_sub,
             condition_tone(predicted)),
        card("Control mode", pretty_mode(mode), "sim", reason),
        card("Valve opening", f"{opening:.1f} %", "sim",
             f"Requested {requested:.1f} %"),
        card("Valve flow", f"{valve_flow:.2f} L/min", "sim",
             f"Demand {desired_flow:.1f} L/min"),
        card("System pressure", f"{pressure:.1f} bar", "sim",
             f"Relief flow {relief_flow:.2f} L/min"),
        card("Replay cycle", str(cycle), "dataset",
             "May not be stable" if unstable else "Stable operation"),
    ]
    st.markdown(card_grid(cards), unsafe_allow_html=True)

    section("Alerts")
    st.markdown(alerts_html(state["alerts"]), unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        section("Dataset labels")
        component_rows = [
            {
                "Component": COMPONENT_TITLES[name],
                "Label": labels[name],
                "Meaning": describe_label(name, labels[name]),
            }
            for name in PROFILE_COLUMNS
            if name in COMPONENT_TITLES
        ]
        st.dataframe(pd.DataFrame(component_rows), hide_index=True)
        st.caption(
            "Labels are used for offline evaluation only. A deployed "
            "system would not have them."
        )
    with right:
        section("Value provenance")
        provenance = [
            ("Valve condition", condition_value,
             SOURCE_LABELS["ml_prediction"]),
            ("Operating condition", "Unstable flag" if unstable else "Stable",
             SOURCE_LABELS["dataset"]),
            ("Desired flow", f"{desired_flow:.1f} L/min",
             SOURCE_LABELS["user_setting"]),
            ("Requested opening", f"{requested:.1f} %",
             SOURCE_LABELS["controller_simulation"]),
            ("Valve opening", f"{opening:.1f} %", SOURCE_LABELS["simulation"]),
            ("Valve flow", f"{valve_flow:.2f} L/min",
             SOURCE_LABELS["simulation"]),
            ("System pressure", f"{pressure:.1f} bar",
             SOURCE_LABELS["simulation"]),
        ]
        st.dataframe(
            pd.DataFrame(provenance, columns=["Quantity", "Value", "Source"]),
            hide_index=True,
        )
    if state["replay_info"]["model_trained_on_this_cycle"]:
        st.info(state["replay_info"]["note"])

# --------------------------------------------------------- valve condition
with tab_condition:
    left, right = st.columns(2)
    with left:
        section("Class probabilities")
        if predicted is None:
            st.warning("No estimate: sensor data invalid (simulated fault).")
        else:
            for label in ("100", "90", "80", "73"):
                p = float(min(1.0, max(0.0, probabilities.get(label, 0.0))))
                meaning = describe_label("valve_condition", int(label))
                st.progress(p, text=f"{label} % ({meaning}): {p:.1%}")
    with right:
        section("Estimate vs dataset label")
        comparison = pd.DataFrame([
            ("ML estimate", "n/a" if predicted is None else f"{predicted} %"),
            ("Dataset label", f"{labels['valve_condition']} %"),
        ], columns=["Source", "Valve condition"])
        st.dataframe(comparison, hide_index=True)
        if predicted is not None:
            agree = predicted == labels["valve_condition"]
            st.caption(
                ("The estimate matches the label. " if agree else
                 "The estimate differs from the label. ")
                + "If this cycle was in the training data, agreement is not "
                "evidence of accuracy; see Model evaluation."
            )

    section("History: dataset label and out-of-fold estimate")
    oof = load_oof_predictions("stable_only")
    if oof is None:
        st.info("Out-of-fold predictions are not available.")
    else:
        start, stop = history_window(n_cycles, cycle, 100)
        st.line_chart(history_frame(twin.targets, oof, start, stop))
        st.caption(
            "Each out-of-fold estimate comes from a model that did not see "
            "that cycle's block of 50 cycles in training. Gaps are cycles "
            "outside the stable-only evaluation (unstable cycles)."
        )

# -------------------------------------------------------------- simulation
with tab_simulation:
    section("Control")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mode", pretty_mode(mode))
    c2.metric("Requested opening", f"{requested:.1f} %")
    c3.metric("Desired flow", f"{desired_flow:.1f} L/min")
    st.caption(reason)

    frame = trace_frame(trace, desired_flow)
    section("Simulated response over one cycle (60 s)")
    st.caption(
        "Simulated values. The run starts with the valve closed and the "
        "plant at its initial pressure. Assumptions are listed under About."
    )
    st.line_chart(frame[OPENING_COLUMNS])
    st.line_chart(frame[FLOW_COLUMNS])
    st.line_chart(frame[PRESSURE_COLUMNS])

    section("Observed and simulated values")
    observed = state["observed_sensors"]
    comparison = pd.DataFrame([
        ("Pressure, PS1 cycle mean", observed["PS1"]["value"], "bar",
         SOURCE_LABELS["dataset"]),
        ("Pressure, simulated plant", pressure, "bar",
         SOURCE_LABELS["simulation"]),
        ("Flow, FS1 cycle mean", observed["FS1"]["value"], "L/min",
         SOURCE_LABELS["dataset"]),
        ("Flow, simulated valve", valve_flow, "L/min",
         SOURCE_LABELS["simulation"]),
    ], columns=["Quantity", "Value", "Unit", "Source"])
    st.dataframe(comparison, hide_index=True)
    st.caption(
        "These are not expected to agree. The simulation uses assumed "
        "parameters and is not calibrated to the test rig."
    )

# ----------------------------------------------------------------- sensors
with tab_sensors:
    section("Observed sensors for this cycle")
    st.caption(
        "Dataset measurements. Each value is the mean over the 60 s cycle."
    )
    st.dataframe(sensor_table(state), hide_index=True)

    section("History")
    chosen_sensor = st.selectbox("Sensor", list(SENSOR_RATES), key="hist_s")
    half_width = st.slider("Cycles shown either side", 20, 300, 100,
                           key="hist_w")
    start, stop = history_window(n_cycles, cycle, half_width)
    st.line_chart(sensor_history(
        twin.features, chosen_sensor, SENSOR_UNITS[chosen_sensor],
        start, stop,
    ))
    st.caption(f"Cycles {start} to {stop - 1}. Selected cycle: {cycle}.")

# ------------------------------------------------------------------- 3D
with tab_3d:
    section("3D schematic")
    st.caption(
        "Generic schematic, not the geometry of the test rig. Only the "
        "simulated opening, flow and pressure move. The actuator is not "
        "simulated. Requires an internet connection (three.js)."
    )
    scene_data = build_scene_data(state, trace, twin.plant)
    show_html(render_html(scene_data), 700)

# -------------------------------------------------------- model evaluation
with tab_evaluation:
    section("Cross-validation")
    st.caption(
        "5-fold grouped, stratified cross-validation. Blocks of 50 "
        "consecutive cycles are kept together. These figures, not the "
        "replay, describe expected accuracy."
    )
    shown_any = False
    for run_name, title in (
        ("stable_only", "Baseline: stable cycles"),
        ("all_cycles", "Experiment: all cycles"),
    ):
        results = load_cv_results(run_name)
        if results is None:
            continue
        shown_any = True
        st.markdown(f"**{title}**")
        r1, r2, r3 = st.columns(3)
        r1.metric("Macro F1",
                  f"{results['macro_f1_mean']:.3f} +/- "
                  f"{results['macro_f1_std']:.3f}")
        r2.metric("Balanced accuracy",
                  f"{results['balanced_accuracy_mean']:.3f} +/- "
                  f"{results['balanced_accuracy_std']:.3f}")
        r3.metric("Cycles", results["n_cycles"])
        left, right = st.columns(2)
        with left:
            st.caption("Confusion matrix (rows: true, columns: predicted)")
            st.dataframe(confusion_frame(results))
        with right:
            st.caption("Per-class results")
            st.dataframe(report_frame(results), hide_index=True)
    if not shown_any:
        st.info("No saved cross-validation results found.")
    else:
        st.warning(
            "The two runs are not comparable. All 756 unstable cycles are "
            "valve class 100, which makes the all-cycles test set easier "
            "and allows a warm-up shortcut."
        )

    comparison_table = load_report_csv("model_comparison.csv")
    if comparison_table is not None:
        section("Model comparison")
        st.dataframe(comparison_table.round(3), hide_index=True)
        st.caption(
            "Default settings, same folds. Differences smaller than the "
            "fold-to-fold spread are not evidence of a better model."
        )

    robustness_table = load_report_csv("robustness_by_component.csv")
    if robustness_table is not None:
        section("Dependence on other components")
        findings = robustness_findings(robustness_table)
        if findings:
            st.warning(
                "Valve accuracy depends on the condition of other "
                "components:\n\n" + "\n".join(f"- {f}" for f in findings)
            )
        st.dataframe(robustness_table.round(3), hide_index=True)

# ------------------------------------------------------------------- about
with tab_about:
    section("Purpose and scope")
    st.write(
        "This application replays recorded cycles from the UCI hydraulic "
        "test rig dataset. It estimates the condition of the proportional "
        "valve with a machine-learning model and passes that estimate to a "
        "simulated valve, a simulated plant and a demonstration controller."
    )
    section("What each value is")
    st.dataframe(pd.DataFrame([
        (SOURCE_LABELS["dataset"],
         "Recorded by the test rig (cycle summaries and condition labels)."),
        (SOURCE_LABELS["ml_prediction"],
         "Estimated by the classifier. An estimate, not a measurement."),
        (SOURCE_LABELS["simulation"],
         "Produced by the virtual valve and plant using assumed parameters."),
        (SOURCE_LABELS["controller_simulation"],
         "Chosen by the demonstration control logic."),
        (SOURCE_LABELS["user_setting"], "Set by the person using this page."),
    ], columns=["Label", "Meaning"]), hide_index=True)

    section("Limitations")
    st.markdown(
        "- The dataset has no valve command or position. Valve position, "
        "demand and plant pressure here are simulated.\n"
        "- One test rig and one dataset. Results do not transfer to other "
        "machines without new data.\n"
        "- Each cycle also has its own cooler, pump and accumulator "
        "condition. Valve accuracy varies with them (see Model "
        "evaluation).\n"
        "- The model was trained on stable cycles only. Estimates for "
        "unstable cycles are outside its training range.\n"
        "- The simulated plant is a simple normalized model, not "
        "calibrated.\n"
        "- This is a software prototype. It commands no equipment and is "
        "not a safety system. Real systems need independent protection "
        "designed and reviewed by qualified engineers."
    )

    section("Simulation assumptions")
    st.dataframe(pd.DataFrame(
        [(c, p, str(v), u) for c, p, v, u in assumption_rows()],
        columns=["Component", "Parameter", "Value", "Unit"],
    ), hide_index=True)

    section("Software")
    st.dataframe(pd.DataFrame([
        ("Application", APP_VERSION),
        ("Python", sys.version.split()[0]),
        ("Streamlit", pkg_version("streamlit")),
        ("pandas", pkg_version("pandas")),
        ("scikit-learn", pkg_version("scikit-learn")),
    ], columns=["Component", "Version"]), hide_index=True)

    section("Data source")
    st.write(
        "Helwig, N., Pignanelli, E., Schuetze, A. (2015). Condition "
        "monitoring of hydraulic systems. UCI Machine Learning Repository. "
        "DOI 10.24432/C5CW21. Licensed under CC BY 4.0. Features in this "
        "application are summary statistics computed from the original "
        "sensor recordings."
    )

st.markdown(
    footer_html(
        "Prototype for demonstration and research. Not for operational "
        f"use. Version {APP_VERSION}."
    ),
    unsafe_allow_html=True,
)