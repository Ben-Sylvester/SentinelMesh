"""
Orchestrator Runner
-------------------
Evaluation runner with:
- Intelligent routing
- Bandit + RL learning
- Escalation
- World model updates

Used to compare against static baseline.
"""

import json
import time
from pathlib import Path
from typing import Dict, List

from core.router import Router
from core.strategy import SingleModelStrategy, ParallelVoteStrategy
from core.explain import build_trace
from adapters.openai import OpenAIAdapter
from adapters.anthropic import AnthropicAdapter
from adapters.mistral import MistralAdapter
from adapters.local import LocalAdapter
from adapters.google import GoogleAdapter


# ----------------------------
# Configuration
# ----------------------------

TASK_DIR = Path("eval/tasks")
OUTPUT_DIR = Path("eval/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = f"orchestrator_{int(time.time())}"


# ----------------------------
# Model Setup
# ----------------------------

openai_gpt4 = OpenAIAdapter("gpt-4.1-mini")
openai_gpt35 = OpenAIAdapter("gpt-3.5-turbo")
claude = AnthropicAdapter("claude-3-haiku")
mistral = MistralAdapter("mistral-small")
local_llama = LocalAdapter("llama3")
google_gemini = GoogleAdapter("gemini-1.5-flash")


# ----------------------------
# Strategies
# ----------------------------

single_openai = SingleModelStrategy(openai_gpt4)
fast_cheap = SingleModelStrategy(mistral)

parallel_ensemble = ParallelVoteStrategy([
    openai_gpt4,
    claude,
    mistral,
    local_llama,
    google_gemini
])

high_accuracy = ParallelVoteStrategy([
    openai_gpt4,
    openai_gpt35,
    google_gemini,
    claude
])

router = Router({
    "single_openai": single_openai,
    "fast_cheap": fast_cheap,
    "parallel_ensemble": parallel_ensemble,
    "high_accuracy": high_accuracy,
})


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
        prompt = item["prompt"]

        # Feature extraction
        features = router.extract_features(prompt)

        # Strategy selection
        strategy = router.select_strategy(features)

        start = time.time()
        result = await strategy.execute(prompt, {})
        latency = time.time() - start

        # Escalation logic
        escalated = False
        if router.should_escalate(result):
            next_strategy = router.escalate(strategy.name)
            if next_strategy:
                escalated = True
                start = time.time()
                result = await next_strategy.execute(prompt, {})
                latency = time.time() - start
                strategy = next_strategy

        # Learning update
        reward = router.update(result)

        total_cost += result.cost_usd

        trace = build_trace(
            features={
                **features.tolist(),
                "task": task_name,
                "reward": reward,
            },
            strategy_name=strategy.name,
            reason="evaluation run",
            result=result
        )

        results.append({
            "task_id": item["id"],
            "prompt": prompt,
            "output": result.output,
            "strategy": strategy.name,
            "reward": round(reward, 4),
            "latency_sec": round(latency, 3),
            "cost_usd": round(result.cost_usd, 6),
            "escalated": escalated,
            "expected_quality": item.get("expected_quality"),
        })

    return {
        "task": task_name,
        "runner": "orchestrator",
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
        "runner": "orchestrator",
        "run_id": RUN_ID,
        "timestamp": time.time(),
        "results": []
    }

    for task_name, items in tasks.items():
        print(f"[ORCH] Running task: {task_name} ({len(items)} prompts)")
        task_result = await run_task(task_name, items)
        summary["results"].append(task_result)

    output_file = OUTPUT_DIR / f"{RUN_ID}.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[ORCH] Results saved to {output_file}")


# ----------------------------
# CLI Execution
# ----------------------------

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
