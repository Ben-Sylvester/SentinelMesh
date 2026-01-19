import json
import requests
from eval.metrics import aggregate

API = "http://localhost:8000/run"
HEADERS = {"x-api-key": "demo-key"}

def run_case(case):
    payload = {
        "prompt": case["prompt"],
        "meta": {
            "benchmark_id": case["id"],
            "expected_type": case.get("expected_type")
        }
    }

    r = requests.post(API, json=payload, headers=HEADERS, timeout=60)
    r.raise_for_status()

    data = r.json()
    trace = data["trace"]

    return {
        "case_id": case["id"],
        "strategy": trace["strategy"],
        "reward": trace["reward"],
        "cost": trace.get("cost_usd", 0.0),
        "latency_ms": trace.get("latency_ms", 0.0),
        "success": trace.get("reward", 0) > 0,
    }

def main():
    cases = json.load(open("eval/benchmark_cases.json"))
    results = []

    print(f"Running {len(cases)} benchmark cases...\n")

    for case in cases:
        try:
            result = run_case(case)
            results.append(result)

            print(
                f"{case['id']} | "
                f"{result['strategy']} | "
                f"reward={result['reward']:.3f} "
                f"cost=${result['cost']:.4f} "
                f"latency={result['latency_ms']:.1f}ms"
            )

        except Exception as e:
            print(f"❌ {case['id']} failed: {e}")

    metrics = aggregate(results)

    print("\n📊 Benchmark Summary")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics

if __name__ == "__main__":
    main()
