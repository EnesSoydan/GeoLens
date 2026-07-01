"""API-level tests for /health, /model-info and /predict.

Booting via ``with TestClient(...)`` runs the lifespan, so the app starts in its
real NOT_READY state. Readiness is simulated by injecting a synthetic
``PredictionService`` + manifest into ``app.state`` — the real serving artifacts
come from the Sprint S1 build.
"""

from __future__ import annotations

import io

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import get_settings
from app.main import create_app
from app.ml.aggregator import DESCRIPTOR_DIM
from app.ml.model_registry import default_manifest
from app.repositories.metadata_repo import InMemoryMetadataRepo, ReferenceMeta
from app.services.prediction import PredictionService
from app.services.retrieval import RetrievalService

_N = 4
_DB_IDS = np.arange(300, 300 + _N, dtype=np.int64)


class _FakeEmbedding:
    def __init__(self, descriptor: np.ndarray) -> None:
        self._descriptor = descriptor

    def embed(self, image):
        return self._descriptor


def _png_bytes(size=(8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size).save(buf, format="PNG")
    return buf.getvalue()


def _ready_service() -> tuple[PredictionService, object]:
    rng = np.random.default_rng(11)
    raw = rng.standard_normal((_N, DESCRIPTOR_DIM)).astype(np.float32)
    vectors = np.ascontiguousarray([v / np.linalg.norm(v) for v in raw], dtype=np.float32)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(DESCRIPTOR_DIM))
    index.add_with_ids(vectors, _DB_IDS)
    retrieval = RetrievalService(index, confidence_threshold=0.5)
    rows = [
        ReferenceMeta(int(_DB_IDS[i]), float(i), float(-i), f"ref-{i}", f"seq-{i}")
        for i in range(_N)
    ]
    service = PredictionService(
        _FakeEmbedding(vectors[0]), retrieval, InMemoryMetadataRepo(rows), "test-v0"
    )
    manifest = default_manifest(index_size=_N)
    return service, manifest


@pytest.fixture
def not_ready_client():
    with TestClient(create_app()) as client:
        yield client


@pytest.fixture
def ready_client():
    with TestClient(create_app()) as client:
        service, manifest = _ready_service()
        client.app.state.prediction_service = service
        client.app.state.manifest = manifest
        client.app.state.model_ready = True
        client.app.state.index_ready = True
        yield client


def test_health_reports_readiness_flags(not_ready_client):
    resp = not_ready_client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["model_ready"] is False
    assert body["index_ready"] is False
    assert "uptime_s" in body


def test_model_info_not_ready_returns_503_envelope(not_ready_client):
    resp = not_ready_client.get("/api/v1/model-info")
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "MODEL_NOT_READY"


def test_predict_not_ready_returns_503_envelope(not_ready_client):
    resp = not_ready_client.post(
        "/api/v1/predict", files={"image": ("q.png", _png_bytes(), "image/png")}
    )
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "MODEL_NOT_READY"


def test_model_info_ready_returns_manifest(ready_client):
    resp = ready_client.get("/api/v1/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["embedding_dim"] == DESCRIPTOR_DIM
    assert body["index_size"] == _N
    assert body["cities"] == list(get_settings().target_cities)


def test_predict_success_returns_predictions(ready_client):
    resp = ready_client.post(
        "/api/v1/predict",
        files={"image": ("q.png", _png_bytes(), "image/png")},
        data={"top_k": "2", "include_heatmap": "true"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 2
    assert body["heatmap_status"] == "disabled"
    assert body["heatmap_png_base64"] is None
    assert body["is_confident"] is True


def test_predict_rejects_unsupported_content_type(ready_client):
    resp = ready_client.post(
        "/api/v1/predict", files={"image": ("q.gif", _png_bytes(), "image/gif")}
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_IMAGE"


def test_predict_rejects_invalid_image(ready_client):
    resp = ready_client.post(
        "/api/v1/predict", files={"image": ("q.png", b"not-an-image", "image/png")}
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_IMAGE"


def test_predict_rejects_oversized_file(ready_client, monkeypatch):
    monkeypatch.setenv("GEOLENS_MAX_UPLOAD_MB", "0")
    get_settings.cache_clear()
    try:
        resp = ready_client.post(
            "/api/v1/predict", files={"image": ("q.png", _png_bytes(), "image/png")}
        )
        assert resp.status_code == 413
        assert resp.json()["error"]["code"] == "FILE_TOO_LARGE"
    finally:
        get_settings.cache_clear()
