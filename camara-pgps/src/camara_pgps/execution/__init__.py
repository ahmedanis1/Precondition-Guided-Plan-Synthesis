"""Execution layer: run a validated plan against the CAMARA API.

    validated PlanResult  ->  APIExecutor  ->  CAMARA API

``execution.mode`` selects the backend: ``mock`` (default), ``rest``, or ``mcp``.
The executor never mutates the plan.
"""

from functools import lru_cache

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import Settings, get_settings
from camara_pgps.execution.executor import APIExecutor, ExecutionError
from camara_pgps.execution.mcp_executor import MCPExecutor
from camara_pgps.execution.mock_executor import MockExecutor
from camara_pgps.execution.rest_executor import RESTExecutor

_EXECUTORS: dict[str, type[APIExecutor]] = {
    "mock": MockExecutor,
    "rest": RESTExecutor,
    "mcp": MCPExecutor,
}


def build_executor(
    settings: Settings | None = None,
    *,
    registry: CamaraRegistry | None = None,
    mode: str | None = None,
) -> APIExecutor:
    """Construct the executor for ``execution.mode`` (or an explicit ``mode``)."""

    settings = settings or get_settings()
    chosen = (mode or settings.execution.mode).lower()
    try:
        executor_cls = _EXECUTORS[chosen]
    except KeyError:
        raise ExecutionError(
            f"unknown execution mode {chosen!r}; expected one of {sorted(_EXECUTORS)}"
        ) from None
    return executor_cls(registry=registry, settings=settings)


@lru_cache(maxsize=1)
def get_executor() -> APIExecutor:
    """Return the process-wide executor from configured settings."""

    return build_executor()


__all__ = [
    "APIExecutor",
    "ExecutionError",
    "MockExecutor",
    "RESTExecutor",
    "MCPExecutor",
    "build_executor",
    "get_executor",
]
