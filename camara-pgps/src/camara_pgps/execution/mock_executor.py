"""Mock executor - the default (``execution.mode = mock``).

Synthesises deterministic, plausible CAMARA responses without any network call.
Response ids are derived with ``uuid5`` so a given plan produces the same facts
every run.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from camara_pgps.execution.executor import APIExecutor
from camara_pgps.models import CamaraOperation, HTTPMethod
from camara_pgps.models.execution import HTTPRequestSpec, StepExecution
from camara_pgps.models.plan import PlanStep

_NAMESPACE = uuid.UUID("f6f0e6a0-0000-4000-8000-000000000000")


def _stable_id(*parts: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, "|".join(parts)))


class MockExecutor(APIExecutor):
    """Deterministic in-memory CAMARA stand-in."""

    mode = "mock"

    def __init__(
        self,
        *,
        responses: Mapping[str, Any] | None = None,
        fail_operations: set[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._responses = dict(responses or {})
        self._fail_operations = set(fail_operations or set())

    def _perform(
        self, request: HTTPRequestSpec, step: PlanStep, operation: CamaraOperation
    ) -> StepExecution:
        now = datetime.now(tz=UTC)
        if operation.operation_id in self._fail_operations:
            return StepExecution(
                step_index=step.index,
                operation_id=operation.operation_id,
                request=request,
                succeeded=False,
                status_code=502,
                error="mock: configured failure",
                started_at=now,
                finished_at=now,
                duration_ms=0.0,
            )

        status_code, body = self._response_for(request, step, operation)
        return StepExecution(
            step_index=step.index,
            operation_id=operation.operation_id,
            request=request,
            succeeded=200 <= status_code < 300,
            status_code=status_code,
            response_body=body,
            response_headers={"content-type": "application/json"},
            started_at=now,
            finished_at=now,
            duration_ms=0.0,
        )

    def _response_for(
        self, request: HTTPRequestSpec, step: PlanStep, operation: CamaraOperation
    ) -> tuple[int, Any]:
        if operation.operation_id in self._responses:
            entry = self._responses[operation.operation_id]
            if isinstance(entry, tuple) and len(entry) == 2:
                return int(entry[0]), entry[1]
            return 200, entry

        args = request.json_body or {}
        session_id = _stable_id(operation.operation_id, str(step.index), "session")
        subscription_id = _stable_id(operation.operation_id, str(step.index), "subscription")
        assignment_id = _stable_id(operation.operation_id, str(step.index), "assignment")
        geofencing_id = _stable_id(operation.operation_id, str(step.index), "geofencing")

        builders: dict[str, Callable[[], tuple[int, Any]]] = {
            "createSession": lambda: (
                201,
                {
                    "sessionId": session_id,
                    "qosStatus": "REQUESTED",
                    "qosProfile": args.get("qosProfile"),
                    "duration": args.get("duration"),
                },
            ),
            "getSession": lambda: (
                200,
                {"sessionId": self._path_id(request) or session_id, "qosStatus": "AVAILABLE"},
            ),
            "deleteSession": lambda: (204, None),
            "extendQosSessionDuration": lambda: (
                200,
                {"sessionId": self._path_id(request) or session_id, "qosStatus": "AVAILABLE"},
            ),
            # Two sessions on purpose (not one) - this is the operation PGPS
            # chains in front of extendQosSessionDuration when sessionId isn't
            # already known, and a single-item list would never exercise the
            # ItemSelectionAgent's job of picking the one that actually
            # matches the user's request (e.g. "my *gaming* session").
            "retrieveSessionsByDevice": lambda: (
                200,
                [
                    {
                        "sessionId": _stable_id(operation.operation_id, str(step.index), "session", "gaming"),
                        "qosStatus": "AVAILABLE",
                        "qosProfile": "QOS_L",
                        "duration": 3600,
                        "applicationServer": {"ipv4Address": "203.0.113.10"},
                        "description": "Low-latency profile typically used for real-time gaming",
                    },
                    {
                        "sessionId": _stable_id(operation.operation_id, str(step.index), "session", "streaming"),
                        "qosStatus": "AVAILABLE",
                        "qosProfile": "QOS_S",
                        "duration": 7200,
                        "applicationServer": {"ipv4Address": "203.0.113.20"},
                        "description": "Standard profile typically used for media streaming",
                    },
                ],
            ),
            # Three profiles on purpose, with genuinely different priority/
            # quality characteristics - this is the operation PGPS chains in
            # front of createSession when qosProfile isn't already known, and
            # a single-item list would never exercise the ItemSelectionAgent's
            # job of picking the one that actually matches the user's request
            # (e.g. "ensure best quality for an interview" should favour the
            # lowest packetDelayBudget / highest priority profile, not just
            # whichever one happens to be listed first).
            "retrieveQoSProfiles": lambda: (
                200,
                [
                    {
                        "name": "QOS_L",
                        "description": "Low-latency profile for real-time, interactive use (video calls, gaming)",
                        "status": "ACTIVE",
                        "priority": 1,
                    },
                    {
                        "name": "QOS_M",
                        "description": "Balanced profile for general browsing and messaging",
                        "status": "ACTIVE",
                        "priority": 50,
                    },
                    {
                        "name": "QOS_S",
                        "description": "Standard profile typically used for media streaming",
                        "status": "ACTIVE",
                        "priority": 80,
                    },
                ],
            ),
            "verifyLocation": lambda: (
                200,
                {"verificationResult": "TRUE", "matchRate": 100},
            ),
            "retrieveLocation": lambda: (
                200,
                {
                    "lastLocationTime": "2026-01-01T00:00:00Z",
                    "area": {"areaType": "CIRCLE", "center": {"latitude": 0, "longitude": 0}},
                },
            ),
            "createQosAssignment": lambda: (
                201,
                {
                    "assignmentId": assignment_id,
                    "status": "AVAILABLE",
                    "qosProfile": args.get("qosProfile"),
                    "device": args.get("device"),
                },
            ),
            "getQosAssignmentById": lambda: (
                200,
                {"assignmentId": self._path_id(request) or assignment_id, "status": "AVAILABLE"},
            ),
            "getQosAssignmentByDevice": lambda: (
                200,
                [{"assignmentId": assignment_id, "status": "AVAILABLE"}],
            ),
            "revokeQosAssignment": lambda: (204, None),
            "createGeofencingSubscription": lambda: (
                201,
                {
                    "id": geofencing_id,
                    "status": "ACTIVE",
                    "protocol": args.get("protocol"),
                    "sink": args.get("sink"),
                    "types": args.get("types"),
                },
            ),
            "retrieveGeofencingSubscription": lambda: (
                200,
                {"id": self._path_id(request) or geofencing_id, "status": "ACTIVE"},
            ),
            "deleteGeofencingSubscription": lambda: (204, None),
            "createInsightsSubscription": lambda: (201, {"subscriptionId": subscription_id}),
            "getInsightsSubscription": lambda: (
                200,
                {"subscriptionId": self._path_id(request) or subscription_id, "status": "ACTIVE"},
            ),
        }
        builder = builders.get(operation.operation_id)
        if builder is not None:
            return builder()

        default_code = 201 if operation.method is HTTPMethod.POST else 200
        return default_code, {"mock": True, "operationId": operation.operation_id}

    @staticmethod
    def _path_id(request: HTTPRequestSpec) -> str | None:
        tail = request.url.rstrip("/").rsplit("/", 1)[-1]
        return tail or None


__all__ = ["MockExecutor"]
