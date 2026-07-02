"""FastAPI dependency providers (inject shared state from ``app.state``).

The lifespan (``app.core.lifespan``) populates ``app.state`` with the serving
runtime. These providers surface it to route handlers and translate the
NOT_READY state into ``ModelNotReadyError`` (503) per the API contract.
"""

from __future__ import annotations

import asyncio

from fastapi import Request

from app.core.exceptions import ModelNotReadyError
from app.ml.model_registry import ModelManifest
from app.services.prediction import PredictionService


def get_prediction_service(request: Request) -> PredictionService:
    """Return the loaded ``PredictionService`` or raise if NOT_READY."""
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        raise ModelNotReadyError()
    return service


def get_manifest(request: Request) -> ModelManifest:
    """Return the loaded index manifest or raise if NOT_READY."""
    manifest = getattr(request.app.state, "manifest", None)
    if manifest is None:
        raise ModelNotReadyError()
    return manifest


def get_gpu_semaphore(request: Request) -> asyncio.Semaphore:
    """Return the process-wide GPU semaphore set up by the lifespan."""
    return request.app.state.gpu_semaphore
