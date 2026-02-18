import logging
from typing import Any

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Minimal agent that delegates every execution decision to the Router.

    Design contract:
    - Agents plan and orchestrate task flow.
    - Agents do NOT select models, retrieval strategy, or escalation paths.
    - All execution goes through router.route() or router.route_agent_task().
    """

    def __init__(self, name: str, router):
        """
        Args:
            name:   Human-readable agent identifier. Used for tagging
                    tasks so the Router can learn per-agent patterns.
            router: A Router instance. The single authority for all
                    strategy, retrieval, and cost/quality decisions.
        """
        self.name = name
        self.router = router

    # -------------------------------------------------
    # Core execution
    # -------------------------------------------------

    async def run(self, task: str) -> Any:
        """
        Delegate a task to SentinelMesh Router.

        Subclasses may override to add pre/post processing, but must
        always funnel execution through self.router.route_agent_task()
        to preserve Router authority.
        """
        logger.debug("Agent '%s' delegating task to Router", self.name)
        result = await self.router.route_agent_task(self.name, task)
        logger.debug("Agent '%s' received result from Router", self.name)
        return result

    # -------------------------------------------------
    # Observability
    # -------------------------------------------------

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"