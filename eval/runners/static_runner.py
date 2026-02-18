"""
Static Runner — baseline evaluation with no routing intelligence.
FIX: strategy.execute() takes one argument (prompt only), not (prompt, {}).
"""
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List

from core.strategy import SingleModelStrategy
from adapters.mistral import MistralAdapter

TASK_DIR   = Path("eval/tasks")
OUTPUT_DIR = Path("eval/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = f"static_{int(time.time())}"

baseline_model    = MistralAdapter("mistral-small")
baseline_strategy = SingleModelStrategy(baseline_model)


def load_tasks() -> Dict[str, List[dict]]:
    tasks = {}
    for file in TASK_DIR.glob("*.json"):
        with open(file) as f:
            tasks[file.stem] = json.load(f)
    return tasks


async def run_task(task_name: str, items: List[dict]) -> dict:
    results    = []
    total_cost = 0.0

    for item in items:
        start  = time.time()
        # FIX: execute() accepts prompt only — no {} context dict
        result = await baseline_strategy.execute(item["prompt"])
        latency = time.time() - start
        total_cost += result.cost_usd
        results.append({
            "task_id":          item["id"],
            "prompt":           item["prompt"],
            "output":           result.output,
            "latency_sec":      round(latency, 3),
            "cost_usd":         round(result.cost_usd, 6),
            "expected_quality": item.get("expected_quality"),
        })

    return {
        "task":           task_name,
        "strategy":       baseline_strategy.name,
        "model":          baseline_model.name,
        "runs":           len(results),
        "total_cost_usd": round(total_cost, 6),
        "results":        results,
    }


async def main():
    tasks   = load_tasks()
    summary = {"runner": "static", "run_id": RUN_ID, "timestamp": time.time(), "results": []}

    for task_name, items in tasks.items():
        print(f"[STATIC] Running task: {task_name} ({len(items)} prompts)")
        summary["results"].append(await run_task(task_name, items))

    output_file = OUTPUT_DIR / f"{RUN_ID}.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[STATIC] Results saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
