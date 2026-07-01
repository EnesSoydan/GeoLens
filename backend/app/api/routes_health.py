"""Health-check route (liveness + readiness)."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.core.config import get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Liveness + readiness payload (see docs/architecture/04-api-sozlesmesi.md)."""

    status: str
    version: str
    model_ready: bool
    index_ready: bool
    uptime_s: int


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Return service liveness, version and artifact readiness."""
    settings = get_settings()
    state = request.app.state
    start_time = getattr(state, "start_time", None)
    uptime_s = int(time.monotonic() - start_time) if start_time is not None else 0
    return HealthResponse(
        status="ok",
        version=settings.version,
        model_ready=getattr(state, "model_ready", False),
        index_ready=getattr(state, "index_ready", False),
        uptime_s=uptime_s,
    )
