"""REST executor - issues real HTTP requests to a CAMARA gateway (``execution.mode = rest``)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from camara_pgps.execution.executor import APIExecutor
from camara_pgps.models import CamaraOperation
from camara_pgps.models.execution import HTTPRequestSpec, StepExecution
from camara_pgps.models.plan import PlanStep


class RESTExecutor(APIExecutor):
    """Executes plan steps as live HTTP calls with ``httpx``."""

    mode = "rest"

    def __init__(self, *, client: Any | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.Client(
                timeout=self.settings.camara.timeout_seconds,
                follow_redirects=True,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def _perform(
        self, request: HTTPRequestSpec, step: PlanStep, operation: CamaraOperation
    ) -> StepExecution:
        started = datetime.now(tz=UTC)
        response = self._get_client().request(
            request.method.value,
            request.url,
            headers=request.headers or None,
            params=request.query_params or None,
            json=request.json_body,
        )
        finished = datetime.now(tz=UTC)

        try:
            body: Any = response.json()
        except ValueError:
            body = response.text or None

        return StepExecution(
            step_index=step.index,
            operation_id=operation.operation_id,
            request=request,
            status_code=response.status_code,
            response_body=body,
            response_headers={k.lower(): v for k, v in response.headers.items()},
            succeeded=200 <= response.status_code < 300,
            error=(
                None
                if 200 <= response.status_code < 300
                else f"HTTP {response.status_code}"
            ),
            started_at=started,
            finished_at=finished,
            duration_ms=(finished - started).total_seconds() * 1000,
        )


__all__ = ["RESTExecutor"]
