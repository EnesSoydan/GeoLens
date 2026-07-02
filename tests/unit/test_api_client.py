"""Unit tests for the Gradio->FastAPI HTTP client, driven by a mock transport (S4)."""

from __future__ import annotations

import httpx
import pytest
from frontend.gradio_app.api_client import GeoLensClient, PredictionError

_OK_BODY = {
    "query_id": "q1",
    "top_prediction": {"lat": 1.0, "lon": 2.0, "similarity": 0.9},
    "predictions": [
        {"rank": 1, "lat": 1.0, "lon": 2.0, "similarity": 0.9, "reference_image_id": "a"},
    ],
    "confidence": 0.9,
    "is_confident": True,
    "heatmap_png_base64": None,
    "heatmap_status": "disabled",
    "model_version": "dinov2_salad",
    "processing_ms": 5,
}


def _client(handler) -> GeoLensClient:
    return GeoLensClient(base_url="http://test", transport=httpx.MockTransport(handler))


def test_predict_returns_parsed_json_on_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/predict"
        assert b"query.png" in request.content
        return httpx.Response(200, json=_OK_BODY)

    body = _client(handler).predict(b"img", "query.png", "image/png", 5, True)
    assert body["top_prediction"]["lat"] == 1.0


def test_predict_raises_on_503_not_ready() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"code": "MODEL_NOT_READY", "message": "x"}})

    with pytest.raises(PredictionError, match="hazır değil"):
        _client(handler).predict(b"img", "q.png", "image/png", 5, False)


def test_predict_surfaces_error_envelope_on_400() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"code": "INVALID_IMAGE", "message": "bozuk"}})

    with pytest.raises(PredictionError, match="INVALID_IMAGE"):
        _client(handler).predict(b"img", "q.png", "image/png", 5, False)


def test_predict_raises_on_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(PredictionError, match="ulaşılamadı"):
        _client(handler).predict(b"img", "q.png", "image/png", 5, False)


def test_predict_sends_top_k_and_heatmap_form_fields() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["content"] = request.content
        return httpx.Response(200, json=_OK_BODY)

    _client(handler).predict(b"img", "q.png", "image/png", 7, True)
    assert b'name="top_k"' in seen["content"] and b"7" in seen["content"]
    assert b'name="include_heatmap"' in seen["content"] and b"true" in seen["content"]
