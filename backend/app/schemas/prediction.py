"""Prediction + model-info response schemas.

Field names/types follow the API contract in
docs/architecture/04-api-sozlesmesi.md. The request side of ``POST /predict`` is
``multipart/form-data`` (see the route), so only response models live here.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.schemas.common import HeatmapStatus


class GeoPoint(BaseModel):
    """A predicted coordinate with its retrieval similarity."""

    lat: float
    lon: float
    similarity: float


class PredictionItem(BaseModel):
    """One ranked retrieval match."""

    rank: int
    lat: float
    lon: float
    similarity: float
    reference_image_id: str
    sequence_id: str | None = None


class PredictionResponse(BaseModel):
    """Full ``POST /predict`` response payload."""

    query_id: str
    top_prediction: GeoPoint
    predictions: list[PredictionItem]
    confidence: float
    is_confident: bool
    heatmap_png_base64: str | None = None
    heatmap_status: HeatmapStatus
    model_version: str
    processing_ms: int


class ModelInfoResponse(BaseModel):
    """``GET /model-info`` payload describing the loaded model + index."""

    backbone: str = "DINOv2 ViT-B/14"
    aggregator: str = "SALAD"
    embedding_dim: int
    index_type: str = "FAISS IndexFlatIP"
    index_size: int
    dataset: str
    model_hash: str | None = None
    cities: list[str]
