import asyncio
import random
import logging
from eval.datasets import load_tasks

logger = logging.getLogger(__name__)


class SelfPlayEngine:
    """
    B-16 fix: previous execute_sync() called asyncio.run() which raises
    RuntimeError("cannot run nested event loop") when called from within
    FastAPI's running event loop.

    SelfPlayEngine is now fully async — episodes run as background tasks.
    Use run_background_async() for integration with a running event loop,
    or run_background() from a CLI context (fresh event loop).
    """

    def __init__(self, router):
        self.router = router
        self.tasks  = load_tasks()

    async def run_episode(self, steps: int = 10) -> None:
        """Run a single synthetic episode (async)."""
        chosen = random.choices(self.tasks, k=steps)
        await asyncio.gather(*[
            self.router.route(task.prompt)
            for task in chosen
        ])

    async def run_background_async(self, rounds: int = 5) -> None:
        """
        Run self-play as a background coroutine inside a running event loop.
        Call via: asyncio.create_task(engine.run_background_async())
        """
        logger.info("Self-play starting: %d rounds", rounds)
        for i in range(rounds):
            await self.run_episode(steps=20)
            logger.info("Self-play round %d/%d complete", i + 1, rounds)
        logger.info("Self-play complete")

    def run_background(self, rounds: int = 5) -> None:
        """Entry point for CLI / script usage (not inside a running loop)."""
        asyncio.run(self.run_background_async(rounds=rounds))
