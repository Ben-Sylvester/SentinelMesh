import asyncio
import logging
from typing import Any, Dict, List, Optional

from core.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AgentExecutor:
    """
    Orchestrates one or more agents against a shared Router.

    Responsibilities:
    - Maintain a registry of named agents.
    - Dispatch tasks to the correct agent.
    - Support concurrent multi-agent execution.
    - Provide observability (last results, agent list).

    The AgentExecutor does NOT make routing decisions.
    Every agent it dispatches must delegate to the Router.
    """

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}

    # -------------------------------------------------
    # Registration
    # -------------------------------------------------

    def register(self, agent: BaseAgent) -> None:
        """Register an agent by its name. Raises on duplicate."""
        if agent.name in self._agents:
            raise ValueError(
                f"An agent named '{agent.name}' is already registered."
            )
        self._agents[agent.name] = agent
        logger.info("AgentExecutor: registered agent '%s'", agent.name)

    def deregister(self, name: str) -> None:
        """Remove an agent by name. No-op if not found."""
        removed = self._agents.pop(name, None)
        if removed:
            logger.info("AgentExecutor: deregistered agent '%s'", name)

    # -------------------------------------------------
    # Single dispatch
    # -------------------------------------------------

    async def run(self, agent_name: str, task: str) -> Any:
        """
        Run a single agent by name.

        Args:
            agent_name: Name of the registered agent to use.
            task:       Task string to pass to the agent.

        Returns:
            The result returned by the agent (and ultimately the Router).

        Raises:
            KeyError if agent_name is not registered.
        """
        agent = self._agents.get(agent_name)
        if agent is None:
            raise KeyError(
                f"No agent named '{agent_name}' is registered. "
                f"Available: {self.list_agents()}"
            )
        logger.info("AgentExecutor: dispatching task to agent '%s'", agent_name)
        return await agent.run(task)

    # -------------------------------------------------
    # Concurrent multi-agent dispatch
    # -------------------------------------------------

    async def run_parallel(
        self,
        tasks: Dict[str, str],
        return_exceptions: bool = True,
    ) -> Dict[str, Any]:
        """
        Dispatch multiple agents concurrently.

        Args:
            tasks:             Mapping of agent_name → task string.
            return_exceptions: If True, exceptions are returned as values
                               instead of propagating (mirrors asyncio.gather).

        Returns:
            Dict of agent_name → result (or Exception if return_exceptions=True).
        """
        agent_names = list(tasks.keys())
        coroutines = [self.run(name, tasks[name]) for name in agent_names]

        results_list = await asyncio.gather(
            *coroutines,
            return_exceptions=return_exceptions,
        )

        results: Dict[str, Any] = {}
        for name, res in zip(agent_names, results_list):
            if isinstance(res, Exception):
                logger.error(
                    "AgentExecutor: agent '%s' raised %s: %s",
                    name, type(res).__name__, res,
                )
            results[name] = res

        return results

    # -------------------------------------------------
    # Sequential pipeline
    # -------------------------------------------------

    async def run_pipeline(
        self,
        steps: List[Dict[str, str]],
    ) -> List[Any]:
        """
        Run agents in sequence, where each step may build on the last.

        Args:
            steps: Ordered list of {"agent": agent_name, "task": task_string}.
                   Tasks are static — for dynamic chaining (passing output
                   of step N to step N+1), call run() directly in a loop.

        Returns:
            Ordered list of results, one per step.
        """
        results = []
        for i, step in enumerate(steps):
            agent_name = step["agent"]
            task = step["task"]
            logger.info(
                "AgentExecutor: pipeline step %d/%d → agent '%s'",
                i + 1, len(steps), agent_name,
            )
            result = await self.run(agent_name, task)
            results.append(result)
        return results

    # -------------------------------------------------
    # Introspection
    # -------------------------------------------------

    def list_agents(self) -> List[str]:
        return list(self._agents.keys())

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        return name in self._agents