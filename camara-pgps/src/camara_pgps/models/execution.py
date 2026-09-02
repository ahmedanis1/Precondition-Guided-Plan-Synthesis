"""Execution models.

Produced by the Execution Layer (milestone 13) when it runs a validated
:class:`~camara_pgps.models.plan.PlanResult` against a CAMARA gateway (or the
bundled mock). One :class:`StepExecution` is recorded per plan step; the
aggregate :class:`ExecutionResult` carries the accumulated fact store.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from camara_pgps.models.api import HTTPMethod


class HTTPRequestSpec(BaseModel):
    """The concrete HTTP request issued for a plan step."""

    method: HTTPMethod
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    query_params: dict[str, Any] = Field(default_factory=dict)
    json_body: Any | None = None


class StepExecution(BaseModel):
    """The outcome of executing a single plan step."""

    step_index: int
    operation_id: str
    request: HTTPRequestSpec | None = None
    status_code: int | None = None
    response_body: Any | None = None
    response_headers: dict[str, str] = Field(default_factory=dict)
    succeeded: bool = False
    error: str | None = None
    extracted_facts: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: float | None = None


class ExecutionResult(BaseModel):
    """The aggregate result of executing a plan."""

    succeeded: bool = False
    steps: list[StepExecution] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def failed_step(self) -> StepExecution | None:
        """Return the first step that did not succeed, if any."""

        return next((s for s in self.steps if not s.succeeded), None)

    def get_step(self, step_index: int) -> StepExecution | None:
        """Return the recorded execution for a given plan-step index."""

        return next((s for s in self.steps if s.step_index == step_index), None)
