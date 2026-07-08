"""Prediction + model-info routes.

``GET /model-info`` reports the loaded model/index manifest. ``POST /predict``
validates the upload, runs the retrieval pipeline under the shared GPU semaphore
and returns ranked geo-predictions. Both return ``503 MODEL_NOT_READY`` until the
serving artifacts are loaded (Sprint S1 build). The Eigen-CAM heatmap is added in
Sprint S3; in S2 ``heatmap_status`` is always ``"disabled"``.
"""

from __future__ import annotations

import asyncio
import io

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageOps, UnidentifiedImageError

from app.api.deps import get_manifest, get_prediction_service
from app.core.config import get_settings
from app.core.exceptions import FileTooLargeError, InvalidImageError
from app.ml.model_registry import ModelManifest
from app.schemas.prediction import ModelInfoResponse, PredictionResponse
from app.services.prediction import PredictionService

router = APIRouter(tags=["predict"])

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


def _strip_exif(image: Image.Image) -> Image.Image:
    """Return an upright, metadata-free copy of ``image``.

    First bake the EXIF orientation into the pixels: phone cameras usually store
    the raw sensor frame (often landscape) plus an Orientation tag that viewers
    apply for display. The MSLS index was built from already-upright images, so a
    query fed in its raw sensor orientation is effectively rotated relative to the
    reference set, yielding a mismatched embedding (low similarity, wrong city).
    ``ImageOps.exif_transpose`` applies that rotation/flip so the model sees the
    image the same way a human does.

    Then rebuild from raw pixel data, which drops every embedded metadata block,
    including EXIF GPS tags. This is a privacy boundary: the service must not
    ingest or persist location data hidden in a user's upload (Faz D risk #27,
    GDPR; see docs/architecture/03-ai-pipeline.md online flow).
    """
    oriented = ImageOps.exif_transpose(image)
    clean = Image.new(oriented.mode, oriented.size)
    clean.putdata(list(oriented.getdata()))
    return clean


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(manifest: ModelManifest = Depends(get_manifest)) -> ModelInfoResponse:
    """Return the loaded model + index description."""
    settings = get_settings()
    return ModelInfoResponse(
        embedding_dim=manifest.embedding_dim,
        index_size=manifest.index_size,
        dataset=manifest.dataset,
        model_hash=manifest.weights_sha256,
        cities=list(settings.target_cities),
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    image: UploadFile = File(...),
    top_k: int | None = Form(default=None),
    include_heatmap: bool = Form(default=True),
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    """Localize an uploaded query image to ranked geographic predictions."""
    settings = get_settings()

    if image.content_type not in _ALLOWED_CONTENT_TYPES:
        raise InvalidImageError(f"Unsupported content type: {image.content_type!r}.")

    raw = await image.read()
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise FileTooLargeError(f"Upload exceeds {settings.max_upload_mb} MB limit.")

    try:
        pil_image: Image.Image = Image.open(io.BytesIO(raw))
        pil_image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("Uploaded file is not a decodable image.") from exc

    # Strip embedded metadata (EXIF GPS) before the image enters the pipeline.
    pil_image = _strip_exif(pil_image)

    resolved_top_k = top_k if top_k is not None else settings.default_top_k

    semaphore: asyncio.Semaphore = request.app.state.gpu_semaphore
    async with semaphore:
        return await run_in_threadpool(
            service.predict, pil_image, resolved_top_k, include_heatmap
        )
