"""
Static Runner
-------------
Baseline evaluation runner with:
- No learning
- No routing intelligence
- No escalation
- Fixed strategy

Used to measure raw model performance without orchestration.
"""

import json
import time
from pathlib import Path
from typing import Dict, List

from core.router import Router
from core.strategy import SingleModelStrategy
from adapters.openai import OpenAIAdapter
from adapters.mistral import MistralAdapter
from adapters.local import LocalAdapter


# ----------------------------
# Configuration
# ----------------------------

TASK_DIR = Path("eval/tasks")
OUTPUT_DIR = Path("eval/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = f"static_{int(time.time())}"


# ----------------------------
# Fixed Strategy Setup
# ----------------------------

# Pick ONE model as baseline
baseline_model = MistralAdapter("mistral-small")
baseline_strategy = SingleModelStrategy(baseline_model)


# ----------------------------
# Task Loading
# ----------------------------

def load_tasks() -> Dict[str, List[dict]]:
    tasks = {}
    for file in TASK_DIR.glob("*.json"):
        with open(file, "r") as f:
            tasks[file.stem] = json.load(f)
    return tasks


# ----------------------------
# Evaluation Loop
# ----------------------------

async def run_task(task_name: str, items: List[dict]) -> dict:
    results = []
    total_cost = 0.0

    for item in items:
        start = time.time()
        result = await baseline_strategy.execute(item["prompt"], {})
        latency = time.time() - start

        total_cost += result.cost_usd

        results.append({
            "task_id": item["id"],
            "prompt": item["prompt"],
            "output": result.output,
            "latency_sec": round(latency, 3),
            "cost_usd": round(result.cost_usd, 6),
            "expected_quality": item.get("expected_quality")
        })

    return {
        "task": task_name,
        "strategy": baseline_strategy.name,
        "model": baseline_model.name,
        "runs": len(results),
        "total_cost_usd": round(total_cost, 6),
        "results": results
    }


# ----------------------------
# Main Entry
# ----------------------------

async def main():
    tasks = load_tasks()
    summary = {
        "runner": "static",
        "run_id": RUN_ID,
        "timestamp": time.time(),
        "results": []
    }

    for task_name, items in tasks.items():
        print(f"[STATIC] Running task: {task_name} ({len(items)} prompts)")
        task_result = await run_task(task_name, items)
        summary["results"].append(task_result)

    output_file = OUTPUT_DIR / f"{RUN_ID}.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[STATIC] Results saved to {output_file}")


# ----------------------------
# CLI Execution
# ----------------------------

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
