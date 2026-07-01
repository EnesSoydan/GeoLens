"""Application exception hierarchy.

A single base ``GeoLensError`` lets FastAPI map domain failures to the standard
error envelope ``{ "error": { "code", "message", "detail"? } }``. Each subclass
pins the HTTP ``status_code`` and the stable ``code`` string defined in the API
contract (docs/architecture/04-api-sozlesmesi.md). Wired to handlers in
``app.main`` (Sprint S2).
"""

from __future__ import annotations


class GeoLensError(Exception):
    """Base class for all GeoLens domain errors."""

    status_code: int = 500
    code: str = "INTERNAL_ERROR"

    def __init__(self, message: str | None = None) -> None:
        """Initialize with an explicit message or fall back to the docstring."""
        resolved = message or (self.__doc__ or "").strip()
        super().__init__(resolved)
        self.message = resolved


class InvalidImageError(GeoLensError):
    """Uploaded file is not a valid or supported image."""

    status_code = 400
    code = "INVALID_IMAGE"


class FileTooLargeError(GeoLensError):
    """Uploaded file exceeds the configured size limit."""

    status_code = 413
    code = "FILE_TOO_LARGE"


class LowConfidenceError(GeoLensError):
    """Prediction confidence is below the configured threshold."""

    # Not an HTTP failure: surfaced as a status field in the response.
    status_code = 200
    code = "LOW_CONFIDENCE"


class ModelNotReadyError(GeoLensError):
    """Model or FAISS index is not loaded, or manifest hash mismatched."""

    status_code = 503
    code = "MODEL_NOT_READY"
