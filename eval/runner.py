import json
import requests
from eval.metrics import aggregate

API     = "http://localhost:8000/run"
HEADERS = {"x-api-key": "demo-key"}


def run_case(case: dict) -> dict:
    payload = {"prompt": case["prompt"]}

    r = requests.post(API, json=payload, headers=HEADERS, timeout=60)
    r.raise_for_status()

    data     = r.json()
    trace    = data["trace"]

    # FIX: ExplainTrace now includes reward field — no more KeyError
    return {
        "case_id":    case["id"],
        "strategy":   trace.get("strategy"),
        "reward":     trace.get("reward", 0.0),   # reward is Optional[float]
        "cost":       trace.get("cost_usd",    0.0),
        "latency_ms": trace.get("latency_ms",  0.0),
        "success":    (trace.get("reward") or 0.0) > 0,
    }


def main():
    cases   = json.load(open("eval/benchmark_cases.json"))
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
        except Exception as exc:
            print(f"❌ {case['id']} failed: {exc}")

    metrics = aggregate(results)

    print("\n📊 Benchmark Summary")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics


if __name__ == "__main__":
    main()
