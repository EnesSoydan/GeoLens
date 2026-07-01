"""Application exception hierarchy.

A single base ``GeoLensError`` lets FastAPI map domain failures to consistent
JSON responses (wired up in Sprint S2). See
docs/architecture/01-repo-ve-backend.md.
"""

from __future__ import annotations


class GeoLensError(Exception):
    """Base class for all GeoLens domain errors."""

    status_code: int = 500

    def __init__(self, message: str | None = None) -> None:
        """Initialize with an explicit message or fall back to the docstring."""
        resolved = message or (self.__doc__ or "").strip()
        super().__init__(resolved)
        self.message = resolved


class InvalidImageError(GeoLensError):
    """Uploaded file is not a valid or supported image."""

    status_code = 422


class LowConfidenceError(GeoLensError):
    """Prediction confidence is below the configured threshold."""

    # Not an HTTP failure: surfaced as a status field in the response.
    status_code = 200


class ModelNotReadyError(GeoLensError):
    """Model or FAISS index is not loaded, or manifest hash mismatched."""

    status_code = 503
