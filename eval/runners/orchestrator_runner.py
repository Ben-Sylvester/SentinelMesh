"""
Orchestrator Runner
-------------------
Evaluation runner that routes through the full SentinelMesh stack:
bandit + RL learning, escalation, world model updates.

FIX: Updated to the current Router API:
  - Router now requires an embedder/retriever as second argument
  - route() replaces old select_strategy() + execute() + update()
  - strategy.execute(prompt) takes one argument (no {} dict)
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List

from core.router import Router
from core.strategy import SingleModelStrategy, ParallelVoteStrategy
from core.retrieval.vector_index import VectorIndex
from core.retrieval.retriever import Retriever
from adapters.openai import OpenAIAdapter
from adapters.anthropic import AnthropicAdapter
from adapters.mistral import MistralAdapter
from adapters.local import LocalAdapter
from adapters.google import GoogleAdapter

TASK_DIR   = Path("eval/tasks")
OUTPUT_DIR = Path("eval/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = f"orchestrator_{int(time.time())}"

# ── Model + Strategy setup ─────────────────────────────────────────────────
openai_gpt4   = OpenAIAdapter("gpt-4.1-mini")
openai_gpt35  = OpenAIAdapter("gpt-3.5-turbo")
claude        = AnthropicAdapter("claude-3-haiku")
mistral       = MistralAdapter("mistral-small")
local_llama   = LocalAdapter("llama3")
google_gemini = GoogleAdapter("gemini-1.5-flash")

single_openai     = SingleModelStrategy(openai_gpt4)
fast_cheap        = SingleModelStrategy(mistral)
parallel_ensemble = ParallelVoteStrategy([openai_gpt4, claude, mistral, local_llama, google_gemini])
high_accuracy     = ParallelVoteStrategy([openai_gpt4, openai_gpt35, google_gemini, claude])

# FIX: Router requires embedder instance as second argument
class _NoOpEmbedder:
    def embed(self, text: str):
        import numpy as np
        return np.zeros(384)

retriever = Retriever(VectorIndex(dim=384), _NoOpEmbedder())

router = Router(
    {
        "single_openai":     single_openai,
        "fast_cheap":        fast_cheap,
        "parallel_ensemble": parallel_ensemble,
        "high_accuracy":     high_accuracy,
    },
    retriever,   # FIX: was missing — caused TypeError
)


def load_tasks() -> Dict[str, List[dict]]:
    tasks = {}
    for file in TASK_DIR.glob("*.json"):
        with open(file) as f:
            tasks[file.stem] = json.load(f)
    return tasks


async def run_task(task_name: str, items: List[dict]) -> dict:
    """
    FIX: uses router.route_with_metadata() — the unified routing path
    that handles selection, retrieval, execution, escalation, and learning.
    Old code called select_strategy() + execute(prompt, {}) + update() separately,
    none of which exist on the refactored Router.
    """
    results    = []
    total_cost = 0.0

    for item in items:
        prompt = item["prompt"]
        start  = time.time()

        result, ctx, reward = await router.route_with_metadata(prompt)
        latency             = time.time() - start

        total_cost += result.cost_usd

        results.append({
            "task_id":          item["id"],
            "prompt":           prompt,
            "output":           result.output,
            "strategy":         ctx.strategy_name,
            "reward":           round(reward, 4),
            "latency_sec":      round(latency, 3),
            "cost_usd":         round(result.cost_usd, 6),
            "selection_source": ctx.selection_source,
            "expected_quality": item.get("expected_quality"),
        })

    return {
        "task":          task_name,
        "runner":        "orchestrator",
        "runs":          len(results),
        "total_cost_usd": round(total_cost, 6),
        "results":       results,
    }


async def main():
    tasks   = load_tasks()
    summary = {
        "runner":    "orchestrator",
        "run_id":    RUN_ID,
        "timestamp": time.time(),
        "results":   [],
    }

    for task_name, items in tasks.items():
        print(f"[ORCH] Running task: {task_name} ({len(items)} prompts)")
        task_result = await run_task(task_name, items)
        summary["results"].append(task_result)

    output_file = OUTPUT_DIR / f"{RUN_ID}.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ORCH] Results saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
