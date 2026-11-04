"""Write pinned requirements files from the packages installed right now.

Pinning matters here: a trained scikit-learn model saved by one version may
not load under another, and Streamlit changes its API between releases.
"""
from importlib.metadata import PackageNotFoundError, version

RUNTIME = ["numpy", "pandas", "scikit-learn", "joblib", "streamlit"]
DEVELOPMENT = ["matplotlib", "pytest"]


def pins(names):
    lines = []
    for name in names:
        try:
            lines.append(f"{name}=={version(name)}")
        except PackageNotFoundError:
            print(f"WARNING: {name} is not installed; left out")
    return lines


runtime = pins(RUNTIME)
development = pins(DEVELOPMENT)

with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(runtime) + "\n")
with open("requirements-dev.txt", "w", encoding="utf-8") as f:
    f.write("-r requirements.txt\n" + "\n".join(development) + "\n")

print("requirements.txt (used by the deployed app):")
print("\n".join("  " + line for line in runtime))
print("requirements-dev.txt (adds tools for development):")
print("\n".join("  " + line for line in development))