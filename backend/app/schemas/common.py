"""Shared schema primitives: the standard error envelope + heatmap status.

The error envelope matches the contract in docs/architecture/04-api-sozlesmesi.md:
``{ "error": { "code", "message", "detail"? } }``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

# Documents why ``heatmap_png_base64`` is populated or null (see API contract).
HeatmapStatus = Literal["included", "skipped_low_confidence", "disabled"]


class ErrorBody(BaseModel):
    """Inner error object of the standard error envelope."""

    code: str
    message: str
    detail: dict[str, Any] | None = None


class ErrorEnvelope(BaseModel):
    """Top-level error response body."""

    error: ErrorBody
