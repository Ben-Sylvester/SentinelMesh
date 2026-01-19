import numpy as np
from typing import Dict
from core.world_model import WorldModel
from core.strategy import Strategy
from core.contextual_bandit import ContextualBandit
from core.reward import compute_reward

# Re-enforcement Learning
from core.rl_policy import RLPolicy
from core.persistence.bandit_store import load_bandit
from core.persistence.world_model_store import load_world_model
from core.persistence.bandit_store import save_bandit
from core.persistence.world_model_store import save_world_model


class Router:
    def __init__(self, strategies: Dict[str, Strategy]):
        self.strategies = strategies
        self.world_model = WorldModel()
        self.rl = RLPolicy()

        self.feature_dim = 5
        self.bandit = ContextualBandit(feature_dim=self.feature_dim)

        for name in strategies:
            self.bandit.register_arm(name)

        load_bandit(self.bandit)
        load_world_model(self.world_model)


        self._last_features = None
        self._last_strategy = None
        self._last_signature = None
        self._belief_strategy = None

    # ----------------------------
    # Feature Engineering
    # ----------------------------

    def extract_features(self, prompt: str) -> np.ndarray:
        """
        Convert prompt → numeric feature vector.
        """
        features = [
            len(prompt) / 500.0,                          # this normalized length
            int("def " in prompt or "class " in prompt),  # code signal
            int("image" in prompt.lower()),               # vision hint
            int("why" in prompt.lower() or "explain" in prompt.lower()),
            prompt.count("?") / 5.0,                      # uncertainty
        ]
        return np.array(features, dtype=float)

    # ----------------------------
    # Strategy Selection
    # ----------------------------

    def select_strategy(self, features) -> Strategy:
        """
        Select a strategy for a given task.

        Priority:
        1. World-model belief
        2. RL policy
        3. Contextual bandit
        """

        signature = self.world_model.task_signature(features)

        belief_strategy = self.world_model.recommend(signature)
        self._belief_strategy = belief_strategy

        rl_choice = None

        # 1️⃣ World-model belief (authoritative if confident)
        if belief_strategy and belief_strategy in self.strategies:
            selected = belief_strategy

            # RL–belief alignment prior (ADVANCED RL)
            self.rl.q[signature][belief_strategy] += 0.05
            source = "belief"

        else:
            # 2️⃣ RL proposal
            actions = list(self.strategies.keys())
            rl_choice = self.rl.select(signature, actions)

            if rl_choice and rl_choice in self.strategies:
                selected = rl_choice
                source = "rl"
            else:
                # 3️⃣ Bandit fallback
                selected = self.bandit.select(features)
                source = "bandit"

        self._last_features = features
        self._last_strategy = selected
        self._last_signature = signature
        self._selection_source = source

        return self.strategies[selected]


    # ----------------------------
    # Learning Feedback
    # ----------------------------

    def update(self, result):
        reward = compute_reward(result)

        # RL update (includes reward normalization)
        self.rl.update(
            state=self._last_signature,
            action=self._last_strategy,
            reward=reward,
            next_state=self._last_signature,
        )

        if self._last_strategy:
            # Bandit learning
            self.bandit.update(
                name=self._last_strategy,
                features=self._last_features,
                reward=reward,
            )

            # World-model belief learning
            self.world_model.update(
                signature=self._last_signature,
                strategy=self._last_strategy,
                reward=reward,
            )

            # Counterfactual updates (cheap simulation)
            for alt_name in self.strategies:
                if alt_name != self._last_strategy:
                    self.world_model.update(
                        signature=self._last_signature,
                        strategy=alt_name,
                        reward=reward * 0.9,
                    )

        # ✅ Persist AFTER learning
        save_bandit(self.bandit)
        save_world_model(self.world_model)

        return reward


    # ----------------------------
    # Escalation Policy
    # ----------------------------

    def escalate(self, current_strategy_name: str):
        escalation_map = {
            "fast_cheap": "single_openai",
            "single_openai": "parallel_ensemble",
        }
        next_name = escalation_map.get(current_strategy_name)
        if not next_name:
            return None
        return self.strategies.get(next_name)

    # ----------------------------
    # Stats / Observability
    # ----------------------------

    def stats(self):
        return self.bandit.stats()

    def last_decision_trace(self):
        return self.bandit.last_scores
