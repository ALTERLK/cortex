"""Per-request context shared between middleware and route handlers.

NOTE (learning): a ContextVar is like a thread-local that also works with
async tasks (and propagates into the threadpool Starlette uses for sync
handlers). The middleware sets the request id once; any log line anywhere
in the request's call stack can read it — no need to thread a request_id
parameter through every function signature.
"""

from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
