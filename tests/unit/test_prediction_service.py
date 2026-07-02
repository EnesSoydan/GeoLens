"""PredictionService unit tests: response assembly over synthetic components.

Uses a fake embedding (returns a fixed descriptor), a synthetic FAISS index and
an in-memory metadata repo. This validates orchestration + payload shaping only;
the real end-to-end flow depends on the Sprint S1 index/metadata build.
"""

from __future__ import annotations

import faiss
import numpy as np
import pytest
from PIL import Image

from app.ml.aggregator import DESCRIPTOR_DIM
from app.repositories.metadata_repo import InMemoryMetadataRepo, ReferenceMeta
from app.services.prediction import PredictionService
from app.services.retrieval import RetrievalService

_N = 5
_DB_IDS = np.arange(200, 200 + _N, dtype=np.int64)


class _FakeEmbedding:
    """Embedding stub returning a fixed descriptor regardless of the image."""

    def __init__(self, descriptor: np.ndarray) -> None:
        self._descriptor = descriptor

    def embed(self, image: Image.Image) -> np.ndarray:
        return self._descriptor


class _FakeXAI:
    """Heatmap stub that records whether it was invoked."""

    def __init__(self) -> None:
        self.calls = 0

    def heatmap_png_base64(self, image: Image.Image) -> str:
        self.calls += 1
        return "FAKE_BASE64_PNG"


def _build_service(query_index: int, threshold: float, xai=None) -> PredictionService:
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((_N, DESCRIPTOR_DIM)).astype(np.float32)
    vectors = np.ascontiguousarray(
        [v / np.linalg.norm(v) for v in raw], dtype=np.float32
    )
    index = faiss.IndexIDMap(faiss.IndexFlatIP(DESCRIPTOR_DIM))
    index.add_with_ids(vectors, _DB_IDS)
    retrieval = RetrievalService(index, confidence_threshold=threshold)

    rows = [
        ReferenceMeta(
            db_id=int(_DB_IDS[i]),
            lat=float(i),
            lon=float(-i),
            reference_image_id=f"ref-{i}",
            sequence_id=f"seq-{i}",
        )
        for i in range(_N)
    ]
    metadata = InMemoryMetadataRepo(rows)
    embedding = _FakeEmbedding(vectors[query_index])
    return PredictionService(embedding, retrieval, metadata, "test-v0", xai=xai)


def test_predict_assembles_ranked_response():
    service = _build_service(query_index=2, threshold=0.5)
    image = Image.new("RGB", (8, 8))

    resp = service.predict(image, top_k=3, include_heatmap=True)

    assert len(resp.predictions) == 3
    top = resp.predictions[0]
    assert top.rank == 1
    assert top.reference_image_id == "ref-2"
    assert top.sequence_id == "seq-2"
    assert resp.top_prediction.lat == 2.0
    assert resp.top_prediction.lon == -2.0
    assert resp.confidence == pytest.approx(1.0, abs=1e-4)
    assert resp.is_confident is True
    assert resp.model_version == "test-v0"
    assert resp.query_id
    assert resp.processing_ms >= 0


def test_predict_low_confidence_still_returns_best():
    # Threshold above the max possible similarity forces is_confident False.
    service = _build_service(query_index=0, threshold=1.5)
    resp = service.predict(Image.new("RGB", (8, 8)), top_k=1, include_heatmap=True)
    assert resp.is_confident is False
    assert len(resp.predictions) == 1


def test_predict_heatmap_disabled_without_xai():
    service = _build_service(query_index=1, threshold=0.5)
    resp = service.predict(Image.new("RGB", (8, 8)), top_k=2, include_heatmap=True)
    assert resp.heatmap_png_base64 is None
    assert resp.heatmap_status == "disabled"


def test_predict_heatmap_included_when_confident_and_requested():
    xai = _FakeXAI()
    service = _build_service(query_index=0, threshold=0.5, xai=xai)
    resp = service.predict(Image.new("RGB", (8, 8)), top_k=1, include_heatmap=True)
    assert resp.heatmap_status == "included"
    assert resp.heatmap_png_base64 == "FAKE_BASE64_PNG"
    assert xai.calls == 1


def test_predict_heatmap_skipped_when_low_confidence():
    xai = _FakeXAI()
    service = _build_service(query_index=0, threshold=1.5, xai=xai)
    resp = service.predict(Image.new("RGB", (8, 8)), top_k=1, include_heatmap=True)
    assert resp.heatmap_status == "skipped_low_confidence"
    assert resp.heatmap_png_base64 is None
    assert xai.calls == 0


def test_predict_heatmap_disabled_when_not_requested():
    xai = _FakeXAI()
    service = _build_service(query_index=0, threshold=0.5, xai=xai)
    resp = service.predict(Image.new("RGB", (8, 8)), top_k=1, include_heatmap=False)
    assert resp.heatmap_status == "disabled"
    assert resp.heatmap_png_base64 is None
    assert xai.calls == 0
