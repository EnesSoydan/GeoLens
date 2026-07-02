"""Application lifespan: shared-state setup + artifact readiness.

On startup the app builds process-wide shared state and decides whether it is
*ready* to serve predictions:

* ``gpu_semaphore`` — the single global ``asyncio.Semaphore(1)`` that serializes
  GPU work (embedding now; Eigen-CAM in S3) to avoid OOM on the 8 GB card
  (see docs/architecture/03-ai-pipeline.md).
* ``start_time`` — monotonic clock for the ``/health`` uptime field.
* readiness slots (``prediction_service``, ``manifest``, ``index_size`` and the
  ``model_ready`` / ``index_ready`` flags).

Loading the *real* model + FAISS index + reference metadata requires the built
artifacts produced by ``scripts/build_index.py`` (Sprint S1, gated on the MSLS
download) and the SQLite metadata repository (also S1). When all three artifacts
(``index.faiss`` + ``manifest.json`` + ``metadata.db``) are present, the loader
shares a single VPR model between the embedding and Eigen-CAM services, wraps the
FAISS index (read via the unicode-safe :mod:`app.ml.faiss_io`) and the SQLite
metadata repo, checks manifest/model compatibility, and serves in a **READY**
state. If any artifact is missing or the manifest is incompatible the app boots
cleanly in **NOT_READY**: ``/health`` reports it, and ``/predict`` / ``/model-info``
return ``503 MODEL_NOT_READY``. This keeps the service (and the test suite)
runnable with no weights download; tests exercise the routes by injecting a
synthetic ``PredictionService`` into ``app.state`` directly.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class _LoadedArtifacts:
    """Wired serving components plus their metadata repo (for teardown)."""

    prediction_service: object
    manifest: object
    index_size: int
    metadata_repo: object


def _load_artifacts(settings: Settings) -> _LoadedArtifacts | None:
    """Load model + index + metadata into a ready ``PredictionService``.

    Returns ``None`` (and logs why) when artifacts are missing or the built index
    manifest is incompatible with the current model, so the app stays NOT_READY.
    Heavy imports (torch, faiss) are local to keep them out of module import time.
    """
    index_path = settings.index_dir / "index.faiss"
    manifest_path = settings.index_dir / "manifest.json"
    db_path = settings.index_dir / "metadata.db"

    missing = [p.name for p in (index_path, manifest_path, db_path) if not p.exists()]
    if missing:
        logger.info(
            "No serving artifacts under %s (missing %s); starting NOT_READY "
            "(build the index with scripts/build_index.py).",
            settings.index_dir,
            ", ".join(missing),
        )
        return None

    from app.ml.faiss_io import read_index
    from app.ml.model_registry import ModelManifest, default_manifest
    from app.repositories.metadata_repo import SqliteMetadataRepo
    from app.services.embedding import EmbeddingService
    from app.services.prediction import PredictionService
    from app.services.retrieval import RetrievalService
    from app.services.xai import XAIService

    manifest = ModelManifest.from_json(manifest_path)
    if not manifest.is_compatible(default_manifest()):
        logger.error(
            "Index manifest at %s is incompatible with the current model "
            "(model/preprocessing drift); staying NOT_READY.",
            manifest_path,
        )
        return None

    device = settings.serving_device
    logger.info("Loading VPR model on %s ...", device)
    # Lazy import: the backbone triggers the torch.hub weight fetch.
    from app.ml.backbone import load_vpr_model

    # One shared model instance feeds both embedding and Eigen-CAM so the GPU
    # holds a single copy of the weights.
    model = load_vpr_model(pretrained=True, device=device)
    embedding = EmbeddingService(model, device=device, image_size=manifest.image_size)
    xai = XAIService(model, image_size=manifest.image_size, device=device)

    index = read_index(index_path)
    retrieval = RetrievalService(index, confidence_threshold=settings.confidence_threshold)
    metadata = SqliteMetadataRepo(db_path)

    prediction_service = PredictionService(
        embedding=embedding,
        retrieval=retrieval,
        metadata=metadata,
        model_version=manifest.model_name,
        xai=xai,
    )
    logger.info(
        "Serving READY: index_size=%d model=%s device=%s.",
        index.ntotal,
        manifest.model_name,
        device,
    )
    return _LoadedArtifacts(
        prediction_service=prediction_service,
        manifest=manifest,
        index_size=int(index.ntotal),
        metadata_repo=metadata,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Set up shared state and (best-effort) load serving artifacts."""
    settings = get_settings()
    app.state.settings = settings
    app.state.start_time = time.monotonic()
    app.state.gpu_semaphore = asyncio.Semaphore(1)

    # Readiness slots. Populated below once the built index + manifest + SQLite
    # metadata exist and the manifest is compatible; NOT_READY otherwise.
    app.state.prediction_service = None
    app.state.manifest = None
    app.state.index_size = 0
    app.state.model_ready = False
    app.state.index_ready = False

    loaded = _load_artifacts(settings)
    if loaded is not None:
        app.state.prediction_service = loaded.prediction_service
        app.state.manifest = loaded.manifest
        app.state.index_size = loaded.index_size
        app.state.model_ready = True
        app.state.index_ready = True

    yield

    if loaded is not None:
        # Release the SQLite connection opened by the metadata repository.
        loaded.metadata_repo.close()  # type: ignore[attr-defined]
