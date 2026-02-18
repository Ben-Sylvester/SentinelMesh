import logging
from typing import Any

logger = logging.getLogger(__name__)


class BaseTool:
    """
    Interface all tools must implement.

    Tools are deterministic execution units — they do NOT decide
    policy, strategy, or retrieval. That authority stays with the Router.
    """

    name: str = "base_tool"
    description: str = ""

    async def execute(self, input_data: str) -> Any:
        """
        Execute the tool with the given input.

        Args:
            input_data: String input produced by the LLM's tool_call.

        Returns:
            Any serialisable result. Callers will str() this before
            feeding it back into a follow-up route() call.
        """
        raise NotImplementedError(
            f"Tool '{self.name}' must implement execute()."
        )

    def __repr__(self) -> str:
        return f"<Tool name={self.name!r} description={self.description!r}>"