"""FastAPI application factory.

Skeleton: exposes ``/health`` so the app is runnable from Sprint S0. The
lifespan (model + FAISS warm-loading, shared GPU semaphore) and the
``/predict`` / ``/model-info`` routes are wired up in Sprint S2.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes_health import router as health_router
from app.core.config import get_settings


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.version)
    app.include_router(health_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
