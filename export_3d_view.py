import argparse
import webbrowser

from src.hydraulic_twin.config import REPORT_DIR
from src.hydraulic_twin.dashboard_data import example_cycles
from src.hydraulic_twin.three_view import build_scene_data, write_html
from src.hydraulic_twin.twin_state import DigitalTwin

parser = argparse.ArgumentParser(
    description="Write a standalone 3D schematic page for one replayed cycle."
)
parser.add_argument("--cycle", type=int, default=None,
                    help="dataset cycle (default: a stable valve-80 cycle)")
parser.add_argument("--flow", type=float, default=8.0,
                    help="desired flow in L/min (default 8.0)")
parser.add_argument("--fault", action="store_true",
                    help="inject a simulated sensor fault")
parser.add_argument("--no-open", action="store_true",
                    help="do not open the page in a browser")
args = parser.parse_args()

twin = DigitalTwin(desired_flow_lpm=args.flow)
cycle = args.cycle
if cycle is None:
    cycle = example_cycles(twin.targets).get("Valve 80 (stable)", 0)

twin.reset_simulation()
state = twin.advance_to_cycle(cycle, simulate_sensor_fault=args.fault)
data = build_scene_data(state, twin.last_trace, twin.plant)

name = f"twin_3d_cycle_{cycle}{'_fault' if args.fault else ''}.html"
path = write_html(REPORT_DIR / name, data)
print("Saved:", path)
print("The page loads three.js from the internet (cdnjs).")
if not args.no_open:
    webbrowser.open(path.resolve().as_uri())