from abc import ABC, abstractmethod
from typing import Optional


class PromptTemplate(ABC):
    name: str

    @abstractmethod
    def format(self, task: str, context: Optional[str] = None) -> str:
        pass
