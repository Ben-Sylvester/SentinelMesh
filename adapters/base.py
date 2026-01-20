from abc import ABC, abstractmethod
from core.models import StrategyResult

# The Adapter Interface
class ModelAdapter(ABC):
    name: str

    @abstractmethod
    async def run(self, prompt: str, context: dict) -> StrategyResult:
        pass
