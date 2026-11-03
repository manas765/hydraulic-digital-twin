import pandas as pd

from src.hydraulic_twin.data_loader import load_profile

BLOCK_SIZE = 50

profile = load_profile()
profile["block"] = profile.index // BLOCK_SIZE

changes = int((profile["valve_condition"].diff().fillna(0) != 0).sum())
print(f"Valve label changes between consecutive cycles: {changes} "
      f"(out of {len(profile) - 1} transitions)")

print(f"\nValve-label counts per block of {BLOCK_SIZE} cycles:")
counts = pd.crosstab(profile["block"], profile["valve_condition"])
counts["unstable_cycles"] = profile.groupby("block")["stable_flag"].sum()
print(counts.to_string())