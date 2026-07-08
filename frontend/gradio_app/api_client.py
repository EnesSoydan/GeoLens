"""HTTP client from the Gradio UI to the FastAPI backend (Sprint S4).

Keeps the UI decoupled from the model: it speaks only the public ``/predict``
contract over HTTP, so the same client works against a local dev server or a
deployed Space (docs/architecture/05-frontend.md). All model/index concerns stay
behind the API; the frontend never imports torch/faiss.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx

# Local-dev fallback: `uvicorn app.main:app` serves the API (and mounted /ui) on
# port 8000. On a deployed Space the API + UI share one container on port 7860, so
# the loopback UI->API call must target that port instead; deploy/entrypoint.sh
# exports GEOLENS_API_BASE_URL=http://127.0.0.1:<serving-port> to override this.
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
PREDICT_PATH = "/api/v1/predict"


def _default_base_url() -> str:
    """Resolve the backend base URL from the environment (dev default 8000)."""
    return os.environ.get("GEOLENS_API_BASE_URL", DEFAULT_BASE_URL)


class PredictionError(Exception):
    """Raised when the backend cannot fulfil a prediction request."""


@dataclass
class GeoLensClient:
    """Thin ``POST /predict`` client returning the parsed response payload.

    ``transport`` is injectable so tests can drive the client with an
    ``httpx.MockTransport`` instead of a live server.
    """

    base_url: str = field(default_factory=_default_base_url)
    timeout: float = 30.0
    transport: httpx.BaseTransport | None = None

    def predict(
        self,
        image_bytes: bytes,
        filename: str,
        content_type: str,
        top_k: int,
        include_heatmap: bool,
    ) -> dict[str, Any]:
        """Send a query image and return the decoded ``PredictionResponse`` dict.

        Raises:
            PredictionError: On transport failure, a NOT_READY (503) backend, or
                any non-200 response (the error envelope message is surfaced).
        """
        url = self.base_url.rstrip("/") + PREDICT_PATH
        files = {"image": (filename, image_bytes, content_type)}
        data = {"top_k": str(top_k), "include_heatmap": str(include_heatmap).lower()}
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                resp = client.post(url, files=files, data=data)
        except httpx.HTTPError as exc:
            raise PredictionError(f"Backend'e ulaşılamadı: {exc}") from exc

        if resp.status_code == 200:
            payload: dict[str, Any] = resp.json()
            return payload
        raise PredictionError(_error_message(resp))


def _error_message(resp: httpx.Response) -> str:
    """Build a user-facing message from an error response's envelope."""
    if resp.status_code == 503:
        return "Model henüz hazır değil (503 MODEL_NOT_READY). İndeks yüklenene kadar bekleyin."
    try:
        error = resp.json().get("error", {})
        code = error.get("code", "ERROR")
        message = error.get("message", resp.text)
        return f"{code}: {message}"
    except ValueError:
        return f"Beklenmeyen yanıt (HTTP {resp.status_code})."
