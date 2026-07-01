"""Prediction + model-info routes.

Skeleton only. ``POST /predict`` (retrieval) is implemented in Sprint S2 and
extended with the Eigen-CAM heatmap + confidence gate in Sprint S3.
``GET /model-info`` (manifest) is implemented in Sprint S2.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["predict"])
