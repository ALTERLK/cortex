"""In-memory request limiting: a per-client sliding window and a daily budget.

Both keep their counters in process memory, which is right for a
single-instance deployment; several instances would share them through
Redis instead.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import UTC, date, datetime


class SlidingWindowLimiter:
    """Allow at most *limit* events per *window_s* seconds per key.

    Keeps the timestamps of recent events per key, drops those that fell out
    of the window, and compares what is left to the limit. Unlike a fixed
    window, a client cannot double its rate by straddling a window boundary.

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


class DailyBudget:
    """Allow at most *limit* events per UTC day, across all clients.

    The per-client limiter stops one client from hammering the API; it does
    nothing against many clients. This is the backstop for a public demo:
    it bounds the number of LLM-backed requests — and therefore the bill —
    for the whole day.

    Args:
        limit: Maximum events per day.
        today: Injectable date source (tests pass a fake).
    """

    def __init__(self, limit: int,
                 today: Callable[[], date] = lambda: datetime.now(UTC).date()) -> None:
        self._limit = limit
        self._today = today
        self._day = today()
        self._used = 0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """Record one event; True if today's budget still had room."""
        with self._lock:
            day = self._today()
            if day != self._day:  # first request of a new day resets the counter
                self._day, self._used = day, 0
            if self._used >= self._limit:
                return False
            self._used += 1
            return True
