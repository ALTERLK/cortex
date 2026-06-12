"""In-memory sliding-window rate limiter.

Good enough for a single-instance deployment; a multi-instance setup would
move the counters to Redis. The algorithm is the honest version of rate
limiting: keep the timestamps of recent events per key, drop the ones that
fell out of the window, compare the remainder to the limit.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable


class SlidingWindowLimiter:
    """Allow at most *limit* events per *window_s* seconds per key.

    Args:
        limit:    Maximum events inside one window.
        window_s: Window length in seconds.
        clock:    Injectable time source (tests pass a fake).
    """

    def __init__(self, limit: int, window_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._limit = limit
        self._window = window_s
        self._clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)
        # Middleware may run in multiple threads (sync handlers in a pool).
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Record an attempt for *key*; True if it fits inside the limit."""
        now = self._clock()
        with self._lock:
            events = self._events[key]
            while events and events[0] <= now - self._window:
                events.popleft()
            if len(events) >= self._limit:
                return False
            events.append(now)
            return True
