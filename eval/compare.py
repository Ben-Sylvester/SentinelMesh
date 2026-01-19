import json
from pathlib import Path

BASELINE = Path("evaluation/results/baseline.json")
LATEST_DIR = Path("evaluation/results")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_latest_run():
    runs = sorted(p for p in LATEST_DIR.glob("run_*.json"))
    if not runs:
        raise RuntimeError("No run files found.")
    return runs[-1]


def compare_metrics(base, new):
    regressions = []

    for b, n in zip(base, new):
        bid = b["id"]

        b_conf = b["metrics"]["confidence"]
        n_conf = n["metrics"]["confidence"]

        b_cost = b["metrics"]["cost_usd"]
        n_cost = n["metrics"]["cost_usd"]

        if n_conf < b_conf - 0.10:
            regressions.append(
                f"{bid}: confidence dropped {b_conf:.2f} → {n_conf:.2f}"
            )

        if n_cost > b_cost * 1.30:
            regressions.append(
                f"{bid}: cost increased ${b_cost:.4f} → ${n_cost:.4f}"
            )

    return regressions


def main():
    baseline = load_json(BASELINE)
    latest_file = find_latest_run()
    latest = load_json(latest_file)

    print(f"\n🔍 Comparing against baseline: {BASELINE.name}")
    print(f"🆕 Latest run: {latest_file.name}\n")

    regressions = compare_metrics(baseline, latest)

    if not regressions:
        print("✅ No regressions detected.")
    else:
        print("❌ Regressions found:\n")
        for r in regressions:
            print(" -", r)


if __name__ == "__main__":
    main()
