import logging
from typing import Dict, List, Optional

from core.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central registry for all available tools.

    Tools are registered once at startup and looked up by name at
    runtime. The registry is read-only during agent execution —
    tools cannot be added or removed mid-flight.
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    # -------------------------------------------------
    # Registration
    # -------------------------------------------------

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool. Raises if a tool with the same name already exists.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"A tool named '{tool.name}' is already registered. "
                "Use a unique name or deregister the existing tool first."
            )
        self._tools[tool.name] = tool
        logger.info("ToolRegistry: registered tool '%s'", tool.name)

    def deregister(self, name: str) -> None:
        """Remove a tool by name. No-op if not found."""
        removed = self._tools.pop(name, None)
        if removed:
            logger.info("ToolRegistry: deregistered tool '%s'", name)

    # -------------------------------------------------
    # Lookup
    # -------------------------------------------------

    def get(self, name: str) -> Optional[BaseTool]:
        """Return the tool or None if not found."""
        tool = self._tools.get(name)
        if tool is None:
            logger.warning("ToolRegistry: unknown tool requested: '%s'", name)
        return tool

    # -------------------------------------------------
    # Introspection
    # -------------------------------------------------

    def list_tools(self) -> List[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def describe_tools(self) -> str:
        """
        Return a human-readable summary of all tools.
        Useful for injecting into system prompts.
        """
        if not self._tools:
            return "No tools are currently registered."
        lines = ["Available tools:"]
        for name, tool in self._tools.items():
            lines.append(f"  - {name}: {tool.description}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools