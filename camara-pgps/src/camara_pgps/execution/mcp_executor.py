"""MCP executor - runs plan steps as Model Context Protocol tool calls
(``execution.mode = mcp``).

Each CAMARA operation is invoked as an MCP tool named ``<tool_prefix><operationId>``
with the resolved arguments. The transport is an injected *session* that exposes
``call_tool(name, arguments)`` (e.g. an ``mcp.ClientSession``); wiring a concrete
CAMARA MCP server is out of scope here, so without a session the executor fails
each step gracefully rather than guessing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from camara_pgps.execution.executor import APIExecutor
from camara_pgps.models import CamaraOperation
from camara_pgps.models.execution import HTTPRequestSpec, StepExecution
from camara_pgps.models.plan import PlanStep


class MCPExecutor(APIExecutor):
    """Executes plan steps via MCP tool calls."""

    mode = "mcp"

    def __init__(
        self,
        *,
        session: Any | None = None,
        tool_prefix: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._session = session
        self._tool_prefix = tool_prefix

    @property
    def configured(self) -> bool:
        return self._session is not None

    def _perform(
        self, request: HTTPRequestSpec, step: PlanStep, operation: CamaraOperation
    ) -> StepExecution:
        now = datetime.now(tz=UTC)
        if self._session is None:
            return StepExecution(
                step_index=step.index,
                operation_id=operation.operation_id,
                request=request,
                succeeded=False,
                error=(
                    "MCP session not configured; provide an MCP session or use "
                    "execution.mode = mock / rest"
                ),
                started_at=now,
                finished_at=now,
            )

        tool_name = f"{self._tool_prefix}{operation.operation_id}"
        arguments = {
            **(request.query_params or {}),
            **(request.json_body or {}),
        }
        started = datetime.now(tz=UTC)
        raw = self._session.call_tool(tool_name, arguments)
        finished = datetime.now(tz=UTC)

        body, is_error = _adapt_tool_result(raw)
        return StepExecution(
            step_index=step.index,
            operation_id=operation.operation_id,
            request=request,
            succeeded=not is_error,
            response_body=body,
            error="MCP tool reported an error" if is_error else None,
            started_at=started,
            finished_at=finished,
            duration_ms=(finished - started).total_seconds() * 1000,
        )


def _adapt_tool_result(raw: Any) -> tuple[Any, bool]:
    """Normalise an MCP ``call_tool`` result to ``(body, is_error)``."""

    is_error = bool(getattr(raw, "isError", False))
    if hasattr(raw, "structuredContent") and raw.structuredContent is not None:
        return raw.structuredContent, is_error
    content = getattr(raw, "content", None)
    if content:
        first = content[0]
        text = getattr(first, "text", None)
        if text is not None:
            return text, is_error
    if isinstance(raw, dict):
        return raw, bool(raw.get("isError", False))
    return raw, is_error


__all__ = ["MCPExecutor"]
