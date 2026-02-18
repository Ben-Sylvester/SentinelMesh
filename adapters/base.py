from abc import ABC, abstractmethod
from core.models import ModelResult


class ModelAdapter(ABC):
    """
    Base interface for all model adapters.

    FIX: return type changed from StrategyResult → ModelResult.
    Adapters are low-level wrappers that return raw inference output.
    Strategy objects (SingleModelStrategy, ParallelVoteStrategy) are
    responsible for assembling ModelResult → StrategyResult, adding
    cost, confidence, raw_outputs, and models_used.
    """
    name: str

    @abstractmethod
    async def run(self, prompt: str, context: dict) -> ModelResult:
        pass
