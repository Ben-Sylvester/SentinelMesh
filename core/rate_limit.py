import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self):
        self.calls = defaultdict(deque)

    def allow(self, tenant_id: str, rpm: int) -> bool:
        now = time.time()
        window = 60

        q = self.calls[tenant_id]

        while q and q[0] < now - window:
            q.popleft()

        if len(q) >= rpm:
            return False

        q.append(now)
        return True
