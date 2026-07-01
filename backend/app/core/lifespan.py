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
download) and the SQLite metadata repository (also S1). Until those exist the app
boots cleanly in a **NOT_READY** state: ``/health`` reports it, and ``/predict``
/ ``/model-info`` return ``503 MODEL_NOT_READY``. This keeps the service (and the
test suite) runnable with no weights download. Tests exercise the routes by
injecting a synthetic ``PredictionService`` into ``app.state`` directly.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Set up shared state and (best-effort) load serving artifacts."""
    settings = get_settings()
    app.state.settings = settings
    app.state.start_time = time.monotonic()
    app.state.gpu_semaphore = asyncio.Semaphore(1)

    # Readiness slots. Populated by the S1 artifact loader once the built index +
    # manifest + SQLite metadata exist; NOT_READY until then.
    app.state.prediction_service = None
    app.state.manifest = None
    app.state.index_size = 0
    app.state.model_ready = False
    app.state.index_ready = False

    index_path = settings.index_dir / "index.faiss"
    manifest_path = settings.index_dir / "manifest.json"
    if index_path.exists() and manifest_path.exists():
        # Artifacts are present but full wiring (model weights + SQLite metadata
        # repository) lands with Sprint S1's build_index. Stay NOT_READY rather
        # than half-load and serve inconsistent results.
        logger.warning(
            "Index artifacts found at %s but runtime loading is completed in "
            "Sprint S1; serving in NOT_READY state.",
            settings.index_dir,
        )
    else:
        logger.info(
            "No serving artifacts under %s; starting in NOT_READY state "
            "(build the index in Sprint S1).",
            settings.index_dir,
        )

    yield

    # No GPU/model resources are held in the S2 skeleton; nothing to tear down.
