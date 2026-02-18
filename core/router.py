import asyncio
import json
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from core.world_model import WorldModel
from core.strategy import Strategy
from core.contextual_bandit import ContextualBandit
from core.reward import compute_reward

from core.rl_policy import RLPolicy
from core.persistence.rl_store import RLStore
from core.persistence.gate_store import GateStore
from core.persistence.meta_store import MetaStore

from core.prompts.registry import PromptRegistry
from core.retrieval.vector_index import VectorIndex
from core.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewardComponents:
    quality_score:   float
    latency_penalty: float
    cost_penalty:    float


@dataclass
class RequestContext:
    features:         np.ndarray
    signature:        object
    strategy_name:    str
    retrieval_flag:   bool
    mode:             str
    selection_source: str
    agent_name:       Optional[str] = None


class MetaPolicy:
    def __init__(self):
        self.modes   = ["cheap", "balanced", "accurate"]
        self.values  = {m: 0.0 for m in self.modes}
        self.epsilon = 0.1
        self.alpha   = 0.1

    def select(self) -> str:
        import random
        if random.random() < self.epsilon:
            return random.choice(self.modes)
        return max(self.values, key=self.values.get)

    def update(self, mode: str, reward: float) -> None:
        self.values[mode] += self.alpha * (reward - self.values[mode])


class RetrievalGate:
    def __init__(self, dim: int):
        self.w  = np.zeros(dim)
        self.lr = 0.01

    def predict(self, x: np.ndarray) -> float:
        z = np.dot(self.w, x)
        return 1 / (1 + np.exp(-z))

    def decide(self, x: np.ndarray, threshold: float = 0.5) -> bool:
        return self.predict(x) > threshold

    def update(self, x: np.ndarray, reward: float) -> None:
        p        = self.predict(x)
        gradient = (reward - p) * x
        self.w  += self.lr * gradient


class Router:

    ESCALATION_MAP: Dict[str, str] = {
        "fast_cheap":    "single_openai",
        "single_openai": "parallel_ensemble",
    }

    def __init__(self, strategies: Dict[str, Strategy], embedder):
        self.strategies = strategies

        self.meta        = MetaPolicy()
        self.world_model = WorldModel()
        self.rl          = RLPolicy()
        self.rl_store    = RLStore()
        self.meta_store  = MetaStore()

        self.prompt_registry = PromptRegistry()

        self.feature_dim  = 5
        self.vector_index = VectorIndex()
        self.retriever    = embedder

        self.retrieval_gate = RetrievalGate(dim=self.feature_dim)

        self.bandit = ContextualBandit(feature_dim=self.feature_dim)
        for name in strategies:
            self.bandit.register_arm(name)

        from core.persistence.bandit_store import load_bandit
        from core.persistence.world_model_store import load_world_model
        self.meta_store.load(self.meta)
        load_bandit(self.bandit)
        load_world_model(self.world_model)
        self.rl.q = self.rl_store.load_rl()

        self.min_epsilon = 0.02
        self.decay_rate  = 0.999

        self._lock = asyncio.Lock()

    # ── Feature Engineering ──────────────────────────────────────────────────

    def extract_features(self, prompt: str) -> np.ndarray:
        length          = len(prompt)
        length_bucket   = 0 if length < 200 else (1 if length < 800 else 2)
        question_count  = prompt.count("?")
        question_bucket = 0 if question_count == 0 else (1 if question_count <= 2 else 2)
        return np.array([
            length_bucket,
            int("def " in prompt or "class " in prompt),
            int("image" in prompt.lower()),
            int("why" in prompt.lower() or "explain" in prompt.lower()),
            question_bucket,
        ], dtype=float)

    # ── Action Selection ─────────────────────────────────────────────────────

    def _select_action(self, features: np.ndarray) -> Tuple[RequestContext, list]:
        signature = self.world_model.task_signature(features)
        mode      = self.meta.select()
        state     = (signature, mode)

        action_space = [
            (name, use_retrieval)
            for name in self.strategies.keys()
            for use_retrieval in (False, True)
        ]

        action = self.rl.select(state, action_space)

        if action not in action_space:
            logger.warning("RLPolicy returned invalid action %r; falling back to bandit.", action)
            strategy_name  = self.bandit.select(features)
            retrieval_flag = False
            source         = "bandit"
        else:
            strategy_name, retrieval_flag = action
            source = "rl"

        ctx = RequestContext(
            features=features,
            signature=signature,
            strategy_name=strategy_name,
            retrieval_flag=retrieval_flag,
            mode=mode,
            selection_source=source,
        )
        return ctx, action_space

    # ── Escalation ───────────────────────────────────────────────────────────

    def escalate(self, current_strategy: str) -> Optional[Strategy]:
        next_name = self.ESCALATION_MAP.get(current_strategy)
        return self.strategies.get(next_name) if next_name else None

    # ── Learning Update (in-memory only, no I/O inside lock) ────────────────

    def _compute_and_apply_update(
        self, ctx: RequestContext, result, latency: Optional[float]
    ) -> Tuple[float, dict]:
        raw        = compute_reward(result, latency)
        components = RewardComponents(*raw)

        reward = (
            0.6 * components.quality_score
            - 0.2 * components.latency_penalty
            - 0.2 * components.cost_penalty
        )

        state  = (ctx.signature, ctx.mode)
        action = (ctx.strategy_name, ctx.retrieval_flag)

        self.rl.update(state=state, action=action, reward=reward, next_state=state)
        self.rl.epsilon = max(self.min_epsilon, self.rl.epsilon * self.decay_rate)

        self.bandit.update_memory(name=ctx.strategy_name, features=ctx.features, reward=reward)

        self.world_model.update(
            signature=ctx.signature,
            strategy=ctx.strategy_name,
            retrieval_flag=ctx.retrieval_flag,
            reward=reward,
        )

        self.retrieval_gate.update(ctx.features, reward)
        self.meta.update(ctx.mode, reward)

        # FIX: use encode_key() so saved keys are round-trippable tuples
        persist_snapshot = {
            "rl_state":  state,   # pass raw tuple; rl_store.save_rl encodes it
            "rl_action": action,
            "rl_value":  float(self.rl.q[state][action]),
        }
        return reward, persist_snapshot

    # ── Persistence (sync, called via thread pool outside lock) ─────────────

    def _persist_sync(self, snapshot: dict) -> None:
        from core.persistence.bandit_store import save_bandit
        from core.persistence.world_model_store import save_world_model
        save_bandit(self.bandit)
        save_world_model(self.world_model)
        self.meta_store.save(self.meta)
        self.rl_store.save_rl(
            snapshot["rl_state"],
            snapshot["rl_action"],
            snapshot["rl_value"],
        )

    async def _persist(self, snapshot: dict) -> None:
        await asyncio.to_thread(self._persist_sync, snapshot)

    # ── Observability ─────────────────────────────────────────────────────────

    def decision_metadata(self, ctx: RequestContext) -> dict:
        return {
            "strategy":         ctx.strategy_name,
            "retrieval_used":   ctx.retrieval_flag,
            "signature":        ctx.signature,
            "mode":             ctx.mode,
            "selection_source": ctx.selection_source,
            "epsilon":          self.rl.epsilon,
            "agent_name":       ctx.agent_name,
            "bandit_scores":    self.bandit.last_scores,
        }

    def stats(self) -> dict:
        return {
            "strategies": list(self.strategies.keys()),
            "bandit":     self.bandit.stats(),
            "rl":         self.rl.stats(),
        }

    # ── Shared routing core ──────────────────────────────────────────────────

    @staticmethod
    def _format_context(chunks) -> Optional[str]:
        """
        FIX: retriever.retrieve() returns a list of dicts. DefaultTemplate
        expects Optional[str]. Convert chunks to a readable text passage
        so the prompt is coherent rather than showing a Python list repr.
        """
        if not chunks:
            return None
        parts = []
        for i, chunk in enumerate(chunks, 1):
            if isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("content") or json.dumps(chunk)
            else:
                text = str(chunk)
            parts.append(f"[{i}] {text}")
        return "\n\n".join(parts)

    async def _route_internal(
        self,
        task: str,
        agent_name: Optional[str] = None,
        template_name: str = "default",
    ):
        features = self.extract_features(task)

        # 1. SELECTION (inside lock)
        async with self._lock:
            ctx, _ = self._select_action(features)
            if agent_name:
                ctx.agent_name = agent_name

        strategy = self.strategies[ctx.strategy_name]

        # 2. RETRIEVAL (outside lock)
        context_str = None
        if ctx.retrieval_flag:
            try:
                chunks      = self.retriever.retrieve(task, k=3)
                # FIX: convert list[dict] → str before passing to template
                context_str = self._format_context(chunks)
            except Exception as exc:
                logger.warning("Retrieval failed, proceeding without context: %s", exc)

        template         = self.prompt_registry.get(template_name)
        formatted_prompt = template.format(task=task, context=context_str)

        # 3. EXECUTION (outside lock)
        escalated = False
        try:
            result  = await strategy.execute(formatted_prompt)
            latency = getattr(result, "latency_ms", None)
        except Exception:
            escalation = self.escalate(ctx.strategy_name)
            if escalation:
                result    = await escalation.execute(formatted_prompt)
                latency   = getattr(result, "latency_ms", None)
                escalated = True
            else:
                raise

        # 4. UPDATE in-memory (inside lock)
        async with self._lock:
            reward, snapshot = self._compute_and_apply_update(ctx, result, latency)

        # 5. PERSIST (outside lock, thread pool)
        await self._persist(snapshot)

        logger.debug(
            "_route_internal | agent=%s strategy=%s escalated=%s reward=%.4f",
            agent_name, ctx.strategy_name, escalated, reward,
        )
        return result, ctx, reward

    # ── Public API ────────────────────────────────────────────────────────────

    async def route(self, task: str, template_name: str = "default"):
        result, _ctx, _reward = await self._route_internal(task=task, template_name=template_name)
        return result

    async def route_agent_task(self, agent_name: str, task: str, template_name: str = "default"):
        tagged_task = f"[AGENT:{agent_name}] {task}"
        result, _ctx, _reward = await self._route_internal(
            task=tagged_task, agent_name=agent_name, template_name=template_name,
        )
        return result

    async def route_with_metadata(self, task: str, template_name: str = "default") -> Tuple:
        return await self._route_internal(task=task, template_name=template_name)
