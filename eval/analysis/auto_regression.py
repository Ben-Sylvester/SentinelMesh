"""
Auto-Regression Benchmark Comparator
------------------------------------
Compares:
- Static runner results
- Orchestrator runner results
- Measures learning trends over time
"""

import json
from pathlib import Path
from statistics import mean

RESULTS_DIR = Path("eval/results")


def load_runs(prefix: str):
    runs = []
    for file in RESULTS_DIR.glob(f"{prefix}_*.json"):
        with open(file) as f:
            runs.append(json.load(f))
    return sorted(runs, key=lambda r: r["timestamp"])


def extract_metrics(run):
    metrics = []
    for task in run["results"]:
        for r in task["results"]:
            metrics.append({
                "reward": r["reward"],
                "latency": r["latency_sec"],
                "cost": r["cost_usd"],
                "strategy": r["strategy"],
            })
    return metrics


def summarize(metrics):
    return {
        "avg_reward": round(mean(m["reward"] for m in metrics), 4),
        "avg_latency": round(mean(m["latency"] for m in metrics), 3),
        "avg_cost": round(mean(m["cost"] for m in metrics), 6),
    }


def trend(runs):
    trendline = []
    for run in runs:
        metrics = extract_metrics(run)
        trendline.append(summarize(metrics))
    return trendline


def main():
    static_runs = load_runs("static")
    orch_runs = load_runs("orchestrator")

    if not static_runs or not orch_runs:
        raise RuntimeError("Missing static or orchestrator runs")

    static_summary = summarize(extract_metrics(static_runs[-1]))
    orch_summary = summarize(extract_metrics(orch_runs[-1]))

    print("\n=== FINAL COMPARISON ===")
    print("Static:", static_summary)
    print("Orchestrator:", orch_summary)

    print("\n=== LEARNING TREND (ORCHESTRATOR) ===")
    for i, point in enumerate(trend(orch_runs)):
        print(f"Run {i+1}: {point}")

    improvement = orch_summary["avg_reward"] - static_summary["avg_reward"]
    print(f"\n📈 Reward Improvement: {round(improvement, 4)}")


if __name__ == "__main__":
    main()

# # Run static benchmark
# python eval/runners/static_runner.py

# # Run orchestrator benchmark (multiple times)
# python eval/runners/orchestrator_runner.py
# python eval/runners/orchestrator_runner.py

# # Compare + regress
# python eval/analysis/auto_regression.py
