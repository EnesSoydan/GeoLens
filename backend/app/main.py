"""FastAPI application factory.

Wires the application together: the lifespan (shared GPU semaphore + artifact
readiness), the routers (``/health``, ``/model-info``, ``/predict``) and the
exception handlers that render the standard error envelope
``{ "error": { "code", "message", "detail"? } }`` (see
docs/architecture/04-api-sozlesmesi.md).
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes_health import router as health_router
from app.api.routes_predict import router as predict_router
from app.core.config import get_settings
from app.core.exceptions import GeoLensError
from app.core.lifespan import lifespan
from app.schemas.common import ErrorBody, ErrorEnvelope


def _envelope(code: str, message: str, detail: dict | None = None) -> dict:
    """Serialize an error into the standard envelope dict."""
    return ErrorEnvelope(error=ErrorBody(code=code, message=message, detail=detail)).model_dump()


async def _geolens_error_handler(request: Request, exc: GeoLensError) -> JSONResponse:
    """Render a domain ``GeoLensError`` as the standard error envelope."""
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(exc.code, exc.message),
    )


async def _validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Render request-validation failures as ``422 VALIDATION_ERROR``."""
    return JSONResponse(
        status_code=422,
        content=_envelope(
            "VALIDATION_ERROR", "Request validation failed.", {"errors": exc.errors()}
        ),
    )


def create_app() -> FastAPI:
    """Build and configure the FastAPI application (API only, no UI)."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan)

    app.add_exception_handler(GeoLensError, _geolens_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_error_handler)  # type: ignore[arg-type]

    app.include_router(health_router, prefix=settings.api_v1_prefix)
    app.include_router(predict_router, prefix=settings.api_v1_prefix)
    return app


def mount_ui(app: FastAPI) -> FastAPI:
    """Mount the Gradio Blocks UI at ``/ui`` on the same app (same container).

    The UI talks to this very app over HTTP (docs/architecture/05-frontend.md), so
    the ``gradio``/``frontend`` imports are local to this function: the API-only
    ``create_app`` (used by the test suite) never pulls in Gradio.
    """
    import gradio as gr
    from frontend.gradio_app.app import build_ui

    return gr.mount_gradio_app(app, build_ui(), path="/ui")


# uvicorn entrypoint ``app.main:app`` serves the API + the mounted Gradio UI.
app = mount_ui(create_app())
