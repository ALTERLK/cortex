"""Tests for the sliding-window rate limiter — fake clock, no sleeping."""

from __future__ import annotations

from cortex.api.ratelimit import SlidingWindowLimiter


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_allows_under_limit() -> None:
    limiter = SlidingWindowLimiter(limit=3, clock=FakeClock())
    assert all(limiter.allow("ip") for _ in range(3))


def test_blocks_over_limit() -> None:
    limiter = SlidingWindowLimiter(limit=2, clock=FakeClock())
    assert limiter.allow("ip")
    assert limiter.allow("ip")
    assert not limiter.allow("ip")


def test_window_expiry_releases_slots() -> None:
    clock = FakeClock()
    limiter = SlidingWindowLimiter(limit=1, window_s=60.0, clock=clock)
    assert limiter.allow("ip")
    assert not limiter.allow("ip")
    clock.t = 61.0  # the old event falls out of the window
    assert limiter.allow("ip")


def test_keys_are_independent() -> None:
    limiter = SlidingWindowLimiter(limit=1, clock=FakeClock())
    assert limiter.allow("alice")
    assert limiter.allow("bob")       # different key, own budget
    assert not limiter.allow("alice")
