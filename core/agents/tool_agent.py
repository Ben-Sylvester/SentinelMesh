import logging
from typing import Any, Optional

from core.agents.base_agent import BaseAgent
from core.tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

# Maximum tool-call rounds per task to prevent infinite loops
MAX_TOOL_ROUNDS = 5


class ToolAgent(BaseAgent):
    """
    Agent that supports a tool-use loop after LLM reasoning.

    Flow per task:
    1. Delegate to Router → get LLM result               (Router decides strategy/retrieval)
    2. If result carries a tool_call, execute the tool.  (Agent handles orchestration)
    3. Feed tool output back through Router for follow-up.(Router again decides strategy)
    4. Repeat up to MAX_TOOL_ROUNDS, then return.

    The Router remains the sole authority for model selection,
    retrieval, escalation, and cost/quality tradeoffs at every step.
    """

    def __init__(self, name: str, router, tool_registry: ToolRegistry):
        super().__init__(name, router)
        self.tool_registry = tool_registry

    # -------------------------------------------------
    # Core execution
    # -------------------------------------------------

    async def run(self, task: str) -> Any:
        """
        Execute a task, resolving any tool calls the LLM emits.
        """
        logger.debug("ToolAgent '%s' starting task", self.name)

        result = await self.router.route_agent_task(self.name, task)

        for round_num in range(1, MAX_TOOL_ROUNDS + 1):

            tool_call = self._extract_tool_call(result)
            if tool_call is None:
                # No tool requested — we're done.
                break

            tool_name = tool_call.get("name")
            tool_input = tool_call.get("input", "")

            tool_output = await self._invoke_tool(tool_name, tool_input)

            if tool_output is None:
                # Unknown tool — stop the loop rather than hallucinating.
                logger.warning(
                    "ToolAgent '%s': tool '%s' not found in registry; "
                    "stopping tool loop at round %d.",
                    self.name, tool_name, round_num,
                )
                break

            # Feed tool output back through SentinelMesh — Router decides next step.
            follow_up = (
                f"Tool '{tool_name}' returned the following output:\n"
                f"{tool_output}\n\n"
                f"Continue the original task using this information."
            )
            logger.debug(
                "ToolAgent '%s': tool round %d — routing follow-up",
                self.name, round_num,
            )
            result = await self.router.route_agent_task(self.name, follow_up)

        else:
            logger.warning(
                "ToolAgent '%s': reached MAX_TOOL_ROUNDS (%d) without "
                "completing task. Returning last result.",
                self.name, MAX_TOOL_ROUNDS,
            )

        return result

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    def _extract_tool_call(self, result) -> Optional[dict]:
        """
        Extract the tool_call dict from a result object, if present.

        Expects result.tool_call to be a dict with keys:
            "name"  (str)  — registered tool name
            "input" (str)  — input string to pass to the tool
        """
        tool_call = getattr(result, "tool_call", None)
        if not tool_call:
            return None
        if not isinstance(tool_call, dict):
            logger.warning(
                "ToolAgent '%s': result.tool_call is %r, expected dict; ignoring.",
                self.name, type(tool_call),
            )
            return None
        if "name" not in tool_call:
            logger.warning(
                "ToolAgent '%s': tool_call missing 'name' key; ignoring.",
                self.name,
            )
            return None
        return tool_call

    async def _invoke_tool(self, tool_name: str, tool_input: str) -> Optional[str]:
        """
        Look up and execute a tool. Returns stringified output or None if not found.
        """
        tool = self.tool_registry.get(tool_name)
        if tool is None:
            return None

        try:
            output = await tool.execute(tool_input)
            return str(output)
        except Exception as exc:
            logger.error(
                "ToolAgent '%s': tool '%s' raised an exception: %s",
                self.name, tool_name, exc,
                exc_info=True,
            )
            # Return error string so the LLM can reason about the failure
            return f"[Tool error] {tool_name} failed with: {exc}"