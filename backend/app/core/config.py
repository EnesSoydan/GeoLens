"""Typed application configuration via Pydantic Settings.

Single source of truth for runtime config: environment variables (prefixed
``GEOLENS_``) and a local ``.env`` file are parsed into a validated, typed
``Settings`` object. See docs/architecture/01-repo-ve-backend.md.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root: backend/app/core/config.py -> parents[3] == project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Validated application settings loaded from env / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GEOLENS_",
        extra="ignore",
    )

    app_name: str = "GeoLens"
    version: str = "0.1.0"
    api_v1_prefix: str = "/api/v1"

    # Data / artifact directories.
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    weights_dir: Path = PROJECT_ROOT / "models" / "weights"

    # MSLS dataset. The official mapillary_sls validation split is the two-city
    # set ["cph", "sf"] (Copenhagen + San Francisco) with public ground truth,
    # which is what the 740-query MSLS-val protocol evaluates on. Amsterdam is a
    # *training* city and has no standard val benchmark, so it is out of scope.
    msls_raw_dir: Path | None = None
    target_cities: tuple[str, ...] = ("cph", "sf")

    @field_validator("target_cities", mode="before")
    @classmethod
    def _split_cities(cls, value: object) -> object:
        """Allow a comma-separated env string (e.g. ``cph,sf``) for the tuple."""
        if isinstance(value, str):
            return tuple(c.strip() for c in value.split(",") if c.strip())
        return value

    # Retrieval / model parameters (some are placeholders, calibrated later).
    embedding_dim: int = 8448  # DINOv2 ViT-B/14 + SALAD.
    default_top_k: int = 5
    confidence_threshold: float = 0.5  # Placeholder - calibrated in Sprint S5.

    # Device the serving model runs on at startup ("cpu" or "cuda"). Defaults to
    # cpu so the app boots anywhere; set GEOLENS_SERVING_DEVICE=cuda on the GPU box.
    serving_device: str = "cpu"

    # API boundary.
    max_upload_mb: int = 10


@lru_cache
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance (single load per process)."""
    return Settings()
