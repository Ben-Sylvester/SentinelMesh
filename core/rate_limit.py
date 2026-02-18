import time
import asyncio
from collections import defaultdict, deque
from typing import Dict


class RateLimiter:
    """
    B-20 fix: previous implementation had two problems:
    1. In-memory only — lost on restart, not shared across workers
    2. No concurrency protection — race condition under async load

    Upgrade:
    - asyncio.Lock per tenant prevents race conditions within a single process
    - Redis backend is supported via the REDIS_URL environment variable —
      when set, the limiter uses Redis INCR+EXPIRE for distributed enforcement
      across multiple workers/processes/restarts.
    - Falls back to an in-memory sliding window when Redis is unavailable,
      with a clear warning in logs.

    Usage with Redis (recommended for production):
        export REDIS_URL=redis://localhost:6379/0
    """

    def __init__(self):
        self._windows: Dict[str, deque]           = defaultdict(deque)
        self._locks:   Dict[str, asyncio.Lock]    = defaultdict(asyncio.Lock)
        self._redis    = None
        self._redis_ok = False
        self._try_connect_redis()

    def _try_connect_redis(self) -> None:
        import os
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            return
        try:
            import redis
            self._redis    = redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            self._redis_ok = True
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Redis unavailable (%s); using in-memory rate limiter. "
                "Rate limits will NOT be shared across workers.",
                exc,
            )

    def allow(self, tenant_id: str, rpm: int) -> bool:
        """
        Synchronous allow check (for use in FastAPI sync endpoints / tests).
        For async endpoints, prefer allow_async().
        """
        if self._redis_ok:
            return self._redis_allow(tenant_id, rpm)
        return self._memory_allow(tenant_id, rpm)

    async def allow_async(self, tenant_id: str, rpm: int) -> bool:
        """Async-safe allow check with per-tenant locking."""
        if self._redis_ok:
            # Redis commands are network I/O — run in thread
            return await asyncio.to_thread(self._redis_allow, tenant_id, rpm)
        async with self._locks[tenant_id]:
            return self._memory_allow(tenant_id, rpm)

    # --------------------------------------------------
    # Backends
    # --------------------------------------------------

    def _redis_allow(self, tenant_id: str, rpm: int) -> bool:
        key     = f"ratelimit:{tenant_id}"
        pipe    = self._redis.pipeline()
        now_sec = int(time.time())
        window  = 60

        pipe.zadd(key,  {str(now_sec): now_sec})
        pipe.zremrangebyscore(key, 0, now_sec - window)
        pipe.zcard(key)
        pipe.expire(key, window + 1)
        _, _, count, _ = pipe.execute()
        return count <= rpm

    def _memory_allow(self, tenant_id: str, rpm: int) -> bool:
        now    = time.time()
        window = 60
        q      = self._windows[tenant_id]

        while q and q[0] < now - window:
            q.popleft()

        if len(q) >= rpm:
            return False

        q.append(now)
        return True
