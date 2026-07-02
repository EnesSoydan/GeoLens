"""PredictionService: orchestrates embed -> search -> metadata -> response.

Pure, synchronous business logic. The route layer owns concurrency concerns
(the global ``gpu_semaphore`` and offloading to a thread pool); this service just
turns a query image into a ``PredictionResponse`` by chaining the embedding,
retrieval and metadata components.

The Eigen-CAM overlay is produced by the optional ``xai`` component, gated by the
API contract: only when the caller requested it *and* the top-1 match is
confident (``resolve_heatmap_status``). When no ``xai`` is injected the service
degrades to ``heatmap_status="disabled"`` with a ``None`` overlay.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from PIL import Image

from app.repositories.metadata_repo import MetadataLookup
from app.schemas.prediction import GeoPoint, PredictionItem, PredictionResponse
from app.services.embedding import EmbeddingService
from app.services.retrieval import RetrievalService, resolve_heatmap_status

if TYPE_CHECKING:
    from app.services.xai import HeatmapGenerator


class PredictionService:
    """Chain embedding + retrieval + metadata into a prediction payload."""

    def __init__(
        self,
        embedding: EmbeddingService,
        retrieval: RetrievalService,
        metadata: MetadataLookup,
        model_version: str,
        xai: HeatmapGenerator | None = None,
    ) -> None:
        """Wire the pipeline components and record the model version string."""
        self._embedding = embedding
        self._retrieval = retrieval
        self._metadata = metadata
        self._model_version = model_version
        self._xai = xai

    def predict(
        self,
        image: Image.Image,
        top_k: int,
        include_heatmap: bool,
    ) -> PredictionResponse:
        """Localize ``image`` and assemble the full prediction response.

        Args:
            image: The query image (already validated at the API boundary).
            top_k: Maximum number of ranked matches to return.
            include_heatmap: Requested XAI flag (ignored in S2; see module docs).

        Returns:
            A ``PredictionResponse`` with ranked matches and confidence signals.
        """
        started = time.perf_counter()
        query_id = str(uuid.uuid4())

        descriptor = self._embedding.embed(image)
        hits = self._retrieval.search(descriptor, top_k)

        items: list[PredictionItem] = []
        for hit in hits:
            meta = self._metadata.get(hit.db_id)
            if meta is None:
                # Index/metadata drift: skip ids with no reference row.
                continue
            items.append(
                PredictionItem(
                    rank=hit.rank,
                    lat=meta.lat,
                    lon=meta.lon,
                    similarity=hit.similarity,
                    reference_image_id=meta.reference_image_id,
                    sequence_id=meta.sequence_id,
                )
            )

        top = items[0]
        confidence = top.similarity
        is_confident = self._retrieval.is_confident(confidence)

        # No overlay capability without an injected XAI component -> "disabled".
        effective_include = include_heatmap and self._xai is not None
        heatmap_status = resolve_heatmap_status(effective_include, is_confident)
        heatmap_png_base64: str | None = None
        if heatmap_status == "included" and self._xai is not None:
            heatmap_png_base64 = self._xai.heatmap_png_base64(image)

        processing_ms = int((time.perf_counter() - started) * 1000)

        return PredictionResponse(
            query_id=query_id,
            top_prediction=GeoPoint(lat=top.lat, lon=top.lon, similarity=top.similarity),
            predictions=items,
            confidence=confidence,
            is_confident=is_confident,
            heatmap_png_base64=heatmap_png_base64,
            heatmap_status=heatmap_status,
            model_version=self._model_version,
            processing_ms=processing_ms,
        )
