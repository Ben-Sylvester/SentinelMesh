"""
Self-Learning System — Progressive LLM Independence

Components:
- ContinuousLearner: Collects training data from every interaction
- ModelBuilder: Trains distilled models on clustered tasks
- DomainAdapter: Detects industry and adapts learning
- IndependenceScheduler: Manages transition to self-sufficiency
- EvolutionEngine: Continuously improves models
"""

from core.learning.continuous_learner import ContinuousLearner, TrainingExample
from core.learning.model_builder import ModelBuilder, SelfModel, ModelConfig
from core.learning.domain_adapter import DomainAdapter, IndustryProfile
from core.learning.independence_scheduler import (
    IndependenceScheduler,
    MaturityLevel,
    IndependenceMetrics,
)
from core.learning.evolution import EvolutionEngine, ModelGeneration

__all__ = [
    "ContinuousLearner",
    "TrainingExample",
    "ModelBuilder",
    "SelfModel",
    "ModelConfig",
    "DomainAdapter",
    "IndustryProfile",
    "IndependenceScheduler",
    "MaturityLevel",
    "IndependenceMetrics",
    "EvolutionEngine",
    "ModelGeneration",
]
