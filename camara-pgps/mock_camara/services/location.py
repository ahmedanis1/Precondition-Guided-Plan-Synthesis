"""Mock Device Location service (verification + retrieval)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["Device Location"])


class VerifyLocationRequest(BaseModel):
    device: Any
    area: Any
    maxAge: int | None = None


class RetrieveLocationRequest(BaseModel):
    device: Any
    maxAge: int | None = None


def _device_hash(device: Any) -> int:
    return int(hashlib.md5(repr(device).encode(), usedforsecurity=False).hexdigest(), 16)


@router.post("/verify")
def verify_location(body: VerifyLocationRequest) -> dict[str, Any]:
    bucket = _device_hash(body.device) % 10
    result = "TRUE" if bucket < 7 else "PARTIAL" if bucket < 9 else "FALSE"
    return {
        "verificationResult": result,
        "matchRate": {"TRUE": 100, "PARTIAL": 60, "FALSE": 0}[result],
        "lastLocationTime": datetime.now(tz=UTC).isoformat(),
    }


@router.post("/retrieve")
def retrieve_location(body: RetrieveLocationRequest) -> dict[str, Any]:
    seed = _device_hash(body.device)
    latitude = round((seed % 18000) / 100 - 90, 4)
    longitude = round((seed // 18000 % 36000) / 100 - 180, 4)
    return {
        "lastLocationTime": datetime.now(tz=UTC).isoformat(),
        "area": {
            "areaType": "CIRCLE",
            "center": {"latitude": latitude, "longitude": longitude},
            "radius": 2000,
        },
    }
