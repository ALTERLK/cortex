"""Per-request context shared between middleware and route handlers.

A ContextVar behaves like a thread-local that also follows async tasks and
propagates into the threadpool Starlette uses for sync handlers. The
middleware sets the request id once; any log line emitted while handling
that request can read it, without passing a request_id through every call.
"""

from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
