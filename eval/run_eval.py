import json
import requests
from datetime import datetime
from pathlib import Path

API_URL = "http://127.0.0.1:8000/run"
BENCHMARK_FILE = Path("evaluation/benchmarks.json")
RESULTS_DIR = Path("evaluation/results")

RESULTS_DIR.mkdir(exist_ok=True)


def load_benchmarks():
    with open(BENCHMARK_FILE) as f:
        return json.load(f)


def run_prompt(prompt: str):
    response = requests.post(API_URL, json={"prompt": prompt}, timeout=120)
    response.raise_for_status()
    return response.json()


def extract_metrics(result: dict):
    trace = result["trace"]
    return {
        "strategy": trace["strategy"],
        "latency_ms": trace["latency_ms"],
        "cost_usd": trace["cost_usd"],
        "confidence": trace["confidence"],
        "reason": trace["reason"],
    }


def main():
    benchmarks = load_benchmarks()
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    all_results = []

    print(f"\n🚀 Running evaluation: {run_id}\n")

    for case in benchmarks:
        print(f"▶ {case['id']}")

        result = run_prompt(case["prompt"])
        metrics = extract_metrics(result)

        record = {
            "id": case["id"],
            "prompt": case["prompt"],
            "output": result["output"],
            "metrics": metrics
        }

        all_results.append(record)

        print("   strategy:", metrics["strategy"])
        print("   confidence:", metrics["confidence"])
        print("   cost: $", metrics["cost_usd"])
        print()

    output_file = RESULTS_DIR / f"run_{run_id}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"✅ Results saved to {output_file}")


if __name__ == "__main__":
    main()
