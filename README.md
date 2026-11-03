# Hydraulic Digital Twin (software prototype)

A software-only prototype that connects **predictive maintenance** with a
**control-oriented digital twin**. It uses the UCI *Condition Monitoring of
Hydraulic Systems* dataset to estimate the condition of a proportional valve
with machine learning, then feeds that estimate into a simulated valve, a
simulated hydraulic plant, and a demonstration controller.

**Status:** Milestones 1-3 complete (data validation, features, valve
classifier, simulation, twin state). The Streamlit dashboard is next.

## What this is, and what it is not

- It **replays historical dataset cycles**. It is not live sensor data.
- The UCI dataset has **no valve command or valve position**. The valve
  position, flow demand, and plant pressure in this project are
  **simulated**, using assumed parameters (listed below).
- It **commands no real equipment** and is not a safety system. Any real
  hydraulic system needs independent protection (for example, a relief valve)
  designed and reviewed by qualified engineers.

Every value in the twin state carries its origin:

| Source tag | Meaning |
|---|---|
| `dataset` | Measured by the UCI rig (cycle summary or label) |
| `ml_prediction` | Estimated by the classifier |
| `simulation` | Produced by the virtual valve and plant |
| `controller_simulation` | Decided by the demonstration controller |
| `user_setting` | Chosen by the person running the demo |

## Dataset

Helwig, N., Pignanelli, E., & Schutze, A. (2015). *Condition monitoring of
hydraulic systems* [Dataset]. UCI Machine Learning Repository.
DOI: [10.24432/C5CW21](https://doi.org/10.24432/C5CW21). Licensed CC BY 4.0.

2,205 cycles of 60 s each, 17 sensors (pressure, flow, temperature, vibration,
motor power, efficiency) at 100, 10, or 1 Hz, plus condition labels for the
cooler, valve, pump, and accumulator. The valve labels are 100 (optimal),
90 (small lag), 80 (severe lag), and 73 (close to total failure).

The dataset is **not included** in this repository.

## Setup (Windows PowerShell)

```powershell
git clone https://github.com/manas765/hydraulic-digital-twin.git
cd hydraulic-digital-twin
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
mkdir data\raw
```

Download the dataset from
https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems
and copy all of its `.txt` files into `data\raw\`.

## Run order

```powershell
python validate_data.py                          # check shapes and labels
python plot_sensors.py                           # figures in reports\
python -m src.hydraulic_twin.preprocessing       # 136 features per cycle
python -m src.hydraulic_twin.train_valve_model   # grouped CV + saves model
python check_model.py                            # model reload check
python -m pytest -q                              # 28 tests
python simulate_open_loop.py                     # valve + plant demo
python simulate_closed_loop.py                   # controller demo
python replay_demo.py                            # full twin state replay
```

Tests that need the processed data or the trained model skip themselves
if those files are missing.

## Machine-learning results

Features: 17 sensors x 8 summary statistics (mean, std, min, max, median,
RMS, range, slope) = 136 per cycle. Model: random forest with balanced class
weights. Evaluation: 5-fold `StratifiedGroupKFold`, with blocks of 50
consecutive cycles kept together, so near-duplicate neighbouring cycles do not
leak between training and test.

Stable cycles only (baseline, 1,449 cycles):

| Metric | Result |
|---|---|
| Macro F1 | 0.863 +/- 0.070 |
| Balanced accuracy | 0.862 +/- 0.070 |

Per-class recall: valve 73 = 0.99, 80 = 0.83, 90 = 0.82, 100 = 0.81. Most
errors are between neighbouring classes, mainly 90 vs 100.

The fold scores ranged from 0.79 to 0.996, so a single split could have been
very misleading.

**Unstable cycles:** all 756 cycles flagged `stable_flag = 1` belong to valve
class 100. A model could learn "looks like warm-up, so class 100" instead of
anything about the valve. The all-cycles experiment (macro F1 0.882 +/- 0.035)
uses a different, easier test set and is **not comparable** with the baseline.
The final model is trained on stable cycles only, and the twin warns when a
replayed cycle is unstable.

**Replay is not an accuracy test.** The final model is trained on every stable
cycle, so replaying one of them only checks the plumbing. Use the
cross-validation numbers above for accuracy.

## Simulation assumptions

All values below are assumed illustration values. They are not taken from the
dataset and are not specifications of any real equipment.

| Component | Parameter | Value |
|---|---|---|
| Valve | Maximum flow | 20 L/min |
| Valve | Response time constant | 0.4 s |
| Valve | Reference pressure drop | 100 bar |
| Plant | Pump flow | 12 L/min |
| Plant | Leak flow | 0.2 L/min |
| Plant | Pressure gain | 0.8 bar per (L/min x s) |
| Plant | Initial pressure | 50 bar |
| Plant | Hard pressure cap | 150 bar |
| Plant | Relief setting | 120 bar |
| Plant | Relief gain | 2.0 L/min per bar |
| Controller | Pressure limits | 5 to 140 bar |
| Controller | Largest accepted demand | 18 L/min |
| Controller | Valve rated flow for conversion | 20 L/min |
| Controller | Opening cap in restricted mode | 25 % |
| Twin | Simulated time per replayed cycle | 60 s (step 0.1 s) |
| Alerts | Low-confidence threshold | top probability < 0.60 |
| Alerts | Flow-shortfall threshold | 1.0 L/min |

Valve health changes only the controller's **mode** (normal, restricted,
fault review). It does not change how the virtual valve moves; that would
need an assumed mapping from labels to response time.

## Project layout

```text
src/hydraulic_twin/
  config.py            paths, sensor rates and units, label values
  data_loader.py       sensor/profile loading and validation
  preprocessing.py     cycle-level feature extraction
  train_valve_model.py grouped cross-validation, final model
  predict.py           loads the saved model, predicts one cycle
  proportional_valve.py  virtual valve (simulation)
  plant_model.py       virtual plant with simulated relief valve
  controller.py        demonstration control logic
  twin_state.py        replay twin: observed + predicted + simulated
tests/                 valve, plant, controller, and twin-state tests
reports/               generated figures
```

## Limitations

- One test rig and one dataset. Results say nothing about other machines.
- Each cycle also has its own cooler, pump, and accumulator condition, so the
  classifier may not isolate valve behaviour.
- The simulated plant is a simple normalized model, not a calibrated one.
- Cycle-level features only. This is a replay of summaries, not a 100 Hz
  stream.

## Planned

- Streamlit dashboard
- PID flow control and further experiments
- Optional 3D visualization