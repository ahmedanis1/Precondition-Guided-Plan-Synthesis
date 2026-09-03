"""Mock Network Insights service (stateful subscriptions)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Response
from pydantic import BaseModel

router = APIRouter(tags=["Network Insights"])

_SUBSCRIPTIONS: dict[str, dict[str, Any]] = {}


def reset() -> None:
    _SUBSCRIPTIONS.clear()


class CreateSubscriptionRequest(BaseModel):
    device: Any
    types: list[str]
    sink: str
    subscriptionExpireTime: str | None = None


@router.post("/subscriptions", status_code=201)
def create_subscription(body: CreateSubscriptionRequest) -> dict[str, Any]:
    subscription_id = str(uuid.uuid4())
    subscription = {
        "subscriptionId": subscription_id,
        "device": body.device,
        "types": body.types,
        "sink": body.sink,
        "status": "ACTIVATION_REQUESTED",
        "startsAt": datetime.now(tz=UTC).isoformat(),
        "expiresAt": body.subscriptionExpireTime,
    }
    _SUBSCRIPTIONS[subscription_id] = subscription
    return subscription


@router.get("/subscriptions/{subscription_id}")
def get_subscription(subscription_id: str, response: Response) -> dict[str, Any]:
    subscription = _SUBSCRIPTIONS.get(subscription_id)
    if subscription is None:
        response.status_code = 404
        return {
            "status": 404,
            "code": "NOT_FOUND",
            "message": f"subscription {subscription_id!r} not found",
        }
    return {**subscription, "status": "ACTIVE"}


@router.get("/subscriptions")
def list_subscriptions() -> list[dict[str, Any]]:
    return list(_SUBSCRIPTIONS.values())
