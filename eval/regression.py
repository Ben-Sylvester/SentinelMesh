import json
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("eval/results")


def generate_baseline():
    df = pd.read_csv(RESULTS_DIR / "benchmark.csv")

    baseline = {
        "avg_quality": float(df["quality"].mean()),
        "avg_cost": float(df["cost"].mean()),
        "avg_latency": float(df["latency_ms"].mean()),
        "avg_reward": float(df["reward"].mean()),
    }

    baseline_file = RESULTS_DIR / "baseline.json"
    baseline_file.write_text(json.dumps(baseline, indent=2))

    print("Baseline saved:")
    print(baseline)


def validate_against_baseline(tolerance: float = 0.05):
    baseline = json.loads((RESULTS_DIR / "baseline.json").read_text())
    df = pd.read_csv(RESULTS_DIR / "benchmark.csv")

    current = {
        "avg_quality": float(df["quality"].mean()),
        "avg_cost": float(df["cost"].mean()),
        "avg_latency": float(df["latency_ms"].mean()),
        "avg_reward": float(df["reward"].mean()),
    }

    print("\nRegression Check:")

    for k in baseline:
        delta = (current[k] - baseline[k]) / max(baseline[k], 1e-6)

        status = "PASS" if abs(delta) <= tolerance else "FAIL"

        print(f"{k}: {current[k]:.4f} vs {baseline[k]:.4f} → {status}")


if __name__ == "__main__":
    generate_baseline()
