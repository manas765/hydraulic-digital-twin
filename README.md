# Hydraulic Digital Twin

![tests](https://github.com/manas765/hydraulic-digital-twin/actions/workflows/tests.yml/badge.svg)

**AI-driven digital twin for predictive monitoring and adaptive control of
hydraulic systems.** A software prototype that estimates the condition of a
proportional valve from recorded sensor data and passes that estimate to a
simulated, safety-aware operating policy.

**Live demo:** https://YOUR-APP-NAME.streamlit.app

> This application replays historical data. It is not live, and it commands
> no real equipment.

<!--
Add screenshots to docs/images/ and uncomment these two lines:
![Overview](docs/images/overview.png)
![3D view](docs/images/3d-view.png)
-->

## What it does

1. **Validates and summarises** the UCI hydraulic test-rig dataset: 2,205
   cycles of 60 s, 17 sensors at 100, 10 and 1 Hz, with condition labels for
   the cooler, valve, pump and accumulator.
2. **Estimates valve condition** (100, 90, 80 or 73 %) from 136 cycle-level
   features with a random-forest classifier, evaluated with grouped,
   stratified cross-validation.
3. **Simulates** a proportional valve (0 to 100 %), a hydraulic plant with a
   relief valve, and a demonstration controller that restricts or inhibits
   operation depending on the estimated valve condition and data validity.
4. **Shows everything in a dashboard** with a 3D schematic. Every value is
   labelled by origin so measurements, estimates and simulation are never
   confused.

| Label | Meaning |
|---|---|
| DATASET | Recorded by the test rig (cycle summaries and condition labels) |
| ML ESTIMATE | Estimated by the classifier. An estimate, not a measurement |
| SIMULATED | Produced by the virtual valve, plant or controller |
| USER SETTING | Chosen by the person using the dashboard |

## Quick start (dashboard only)

The processed features and the trained model are included, so the dashboard
runs without downloading the 700 MB raw dataset. Python 3.11 is recommended
because the dependency versions are pinned.

```bash
git clone https://github.com/manas765/hydraulic-digital-twin.git
cd hydraulic-digital-twin
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

If the saved model cannot be loaded (for example because of a different
scikit-learn version), it is retrained automatically from the processed data.

## Reproduce everything from the raw data

Download the dataset from the
[UCI repository](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)
and copy its `.txt` files into `data/raw/`. Then:

```bash
pip install -r requirements-dev.txt
python validate_data.py                          # shapes, labels, alignment
python plot_sensors.py                           # sensor figures in reports/
python -m src.hydraulic_twin.preprocessing       # 136 features per cycle
python -m src.hydraulic_twin.train_valve_model   # grouped CV + saved model
python -m src.hydraulic_twin.train_valve_model --include-unstable
python run_ml_experiments.py                     # model comparison, robustness
python run_model_selection.py --seeds 5          # repeated CV, sensor ablation
python -m pytest -q                              # 59 tests
```

Other entry points: `simulate_open_loop.py`, `simulate_closed_loop.py`,
`replay_demo.py` and `export_3d_view.py` (standalone 3D page).

## Results

Cross-validation on the 1,449 stable cycles (5 folds, blocks of 50
consecutive cycles kept together, seed 42):

| Metric | Result |
|---|---|
| Macro F1 | 0.863 +/- 0.070 |
| Balanced accuracy | 0.862 +/- 0.070 |

| Valve condition | Precision | Recall |
|---|---|---|
| 100 % (optimal) | 0.85 | 0.81 |
| 90 % (small lag) | 0.78 | 0.82 |
| 80 % (severe lag) | 0.94 | 0.83 |
| 73 % (close to total failure) | 0.90 | 0.99 |

Things worth knowing before relying on these numbers:

- **Scores vary a lot between folds** (macro F1 from 0.79 to 0.996), so a
  single random split would have been misleading.
- **All 756 unstable cycles belong to valve class 100.** The all-cycles run
  (macro F1 0.882 +/- 0.035) uses an easier test set and is not comparable
  with the baseline. The deployed model is trained on stable cycles only.
- **Accuracy depends on the other components.** Valve accuracy is 0.70 when
  the cooler is near total failure, against 0.96 and 0.93 for the other
  cooler states. The dashboard reports this under Model evaluation.
- **Model families:** with default settings and one fold assignment, gradient
  boosting scored 0.906 +/- 0.044 and scaled logistic regression 0.869 +/-
  0.051, against 0.863 +/- 0.070 for the random forest. Differences smaller
  than the fold-to-fold spread are not evidence, so the random forest is kept.
- **Replaying a stable cycle is not an accuracy test.** The deployed model
  was trained on those cycles.

## Simulation assumptions

The dataset has no valve command or position, so these values are assumed
for demonstration. They are not calibrated against hardware. The dashboard's
About tab shows the same table, generated from the code.

| Component | Parameter | Value |
|---|---|---|
| Valve | Maximum flow | 20 L/min |
| Valve | Response time constant | 0.4 s |
| Valve | Reference pressure drop | 100 bar |
| Plant | Pump flow | 12 L/min |
| Plant | Leak flow | 0.2 L/min |
| Plant | Pressure gain | 0.8 bar per (L/min s) |
| Plant | Initial pressure | 50 bar |
| Plant | Hard pressure cap | 150 bar |
| Plant | Relief setting / gain | 120 bar / 2.0 L/min per bar |
| Controller | Pressure limits | 5 to 140 bar |
| Controller | Largest accepted demand | 18 L/min |
| Controller | Valve rated flow (demand conversion) | 20 L/min |
| Controller | Opening cap in restricted mode | 25 % |

With these assumptions the simulated pump can deliver about 11.8 L/min to the
valve, so demands above that cannot be met whatever the controller does.

## Limitations

- One test rig and one dataset. Results do not transfer to other machines
  without new data.
- Valve position, demand and plant pressure are simulated, not recorded.
- The simulated plant is a simple normalised model.
- Valve health changes the controller's mode only. It does not change how the
  simulated valve moves, because that would need an assumed degradation model.
- Control experiments are demonstrated but not quantified (no formal
  tracking-error or settling-time study, no fixed-opening comparison).

## Safety and hardware boundary

This is a software prototype. It is not a safety system and must not be used
to command equipment. Closing the valve on a fault is acceptable in the
simulation only because the simulated plant has a relief valve; on real
equipment the safe state must come from a hazard analysis.

A hardware version would need, at least: the valve datasheet and driver
interface, valve and actuator position feedback, pressure and flow sensors at
known locations, pump and relief-valve characteristics, fluid properties,
hydraulic volume and line layout, an industrial controller and communication
interface, and independent emergency stop and pressure protection. Start
read-only, and consider actuation only after independent review of the plant
model, control design, risk assessment and interlocks.

## Repository layout

```text
dashboard/app.py               Streamlit application
src/hydraulic_twin/
  config.py                    paths, sensor rates and units, label values
  data_loader.py               sensor and profile loading, validation
  preprocessing.py             cycle-level feature extraction
  train_valve_model.py         grouped cross-validation, final model
  predict.py                   model loading (retrains if it cannot load)
  ml_experiments.py            model comparison, robustness by component
  model_selection.py           repeated CV, sensor-group ablation helpers
  proportional_valve.py        virtual valve
  plant_model.py               virtual plant with relief valve
  controller.py                demonstration control logic
  twin_state.py                replay twin with value provenance
  dashboard_data.py            dashboard helper functions
  ui_components.py             theme-independent HTML components
  three_view.py                builds the 3D page
  assets/twin_3d_template.html three.js scene
tests/                         59 tests
data/processed/                features and targets (included)
models/                        trained model and feature list (included)
reports/                       evaluation results and figures
docs/                          demo script and project report
.streamlit/config.toml         theme
Dockerfile                     container image
.github/workflows/tests.yml    continuous integration
```

## Tests and continuous integration

`python -m pytest -q` runs 59 tests covering the virtual valve, plant,
controller, twin state, dashboard helpers, UI components, the 3D page builder,
model loading and the experiment helpers. The same command runs on every push
through GitHub Actions.

## Deployment

**Streamlit Community Cloud:** choose this repository, branch `main`, main
file `dashboard/app.py`, and Python 3.11 under Advanced settings. The app
hibernates after a period without traffic, so the first visit may be slow. The
3D view loads three.js from a CDN and needs an internet connection.

**Container:**

```bash
docker build -t hydraulic-twin .
docker run -p 8501:8501 hydraulic-twin
```

## Data source and licence

Helwig, N., Pignanelli, E., Schuetze, A. (2015). *Condition monitoring of
hydraulic systems.* UCI Machine Learning Repository.
DOI: [10.24432/C5CW21](https://doi.org/10.24432/C5CW21). Licensed under
CC BY 4.0. The processed features in `data/processed/` are summary statistics
computed from the original sensor recordings.

No licence has been chosen for the source code yet, so by default all rights
are reserved. Add a `LICENSE` file before others reuse it.