"""Mock Quality on Demand service.

Stateful in-memory QoS sessions: ``createSession`` -> ``getSession`` ->
``deleteSession`` behave like a real resource lifecycle.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field

router = APIRouter(tags=["Quality on Demand"])

_SESSIONS: dict[str, dict[str, Any]] = {}


def reset() -> None:
    _SESSIONS.clear()


class CreateSessionRequest(BaseModel):
    device: Any
    applicationServer: Any | None = None
    qosProfile: str
    duration: int = Field(gt=0)
    devicePorts: Any | None = None
    applicationServerPorts: Any | None = None
    webhook: Any | None = None


@router.post("/sessions", status_code=201)
def create_session(body: CreateSessionRequest) -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    session = {
        "sessionId": session_id,
        "device": body.device,
        "applicationServer": body.applicationServer,
        "qosProfile": body.qosProfile,
        "duration": body.duration,
        "qosStatus": "REQUESTED",
        "startedAt": datetime.now(tz=UTC).isoformat(),
    }
    _SESSIONS[session_id] = session
    return session


@router.get("/sessions/{session_id}")
def get_session(session_id: str, response: Response) -> dict[str, Any]:
    session = _SESSIONS.get(session_id)
    if session is None:
        response.status_code = 404
        return {
            "status": 404,
            "code": "NOT_FOUND",
            "message": f"session {session_id!r} not found",
        }
    return {**session, "qosStatus": "AVAILABLE"}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, response: Response) -> dict[str, Any] | None:
    if _SESSIONS.pop(session_id, None) is None:
        response.status_code = 404
        return {
            "status": 404,
            "code": "NOT_FOUND",
            "message": f"session {session_id!r} not found",
        }
    response.status_code = 204
    return None


@router.get("/sessions")
def list_sessions() -> list[dict[str, Any]]:
    return list(_SESSIONS.values())
