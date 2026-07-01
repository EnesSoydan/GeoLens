"""Health-check route."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Liveness payload."""

    status: str
    version: str


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service liveness and version."""
    settings = get_settings()
    return HealthResponse(status="ok", version=settings.version)
