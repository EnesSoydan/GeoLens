"""Unit tests for the data/GPU-independent frontend rendering helpers (S4)."""

from __future__ import annotations

import base64
import io

from frontend.gradio_app.rendering import (
    TABLE_HEADERS,
    build_map_html,
    confidence_label,
    decode_heatmap,
    predictions_to_rows,
)
from PIL import Image


def _png_base64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _response(*, is_confident: bool = True, heatmap: str | None = None) -> dict:
    return {
        "query_id": "q1",
        "top_prediction": {"lat": 55.69, "lon": 12.56, "similarity": 0.72},
        "predictions": [
            {"rank": 1, "lat": 55.69, "lon": 12.56, "similarity": 0.72, "reference_image_id": "a"},
            {"rank": 2, "lat": 55.70, "lon": 12.57, "similarity": 0.61, "reference_image_id": "b"},
        ],
        "confidence": 0.72,
        "is_confident": is_confident,
        "heatmap_png_base64": heatmap,
        "heatmap_status": "included" if heatmap else "disabled",
        "model_version": "dinov2_salad",
        "processing_ms": 12,
    }


def test_predictions_to_rows_matches_headers() -> None:
    rows = predictions_to_rows(_response()["predictions"])
    assert len(TABLE_HEADERS) == 5
    assert rows[0] == [1, 55.69, 12.56, 0.72, "a"]
    assert rows[1][0] == 2


def test_build_map_html_contains_coordinates() -> None:
    html = build_map_html(_response()["predictions"])
    assert "<iframe" in html or "folium" in html.lower()
    assert "12.56" in html and "55.69" in html


def test_build_map_html_empty_predictions_is_placeholder() -> None:
    html = build_map_html([])
    assert "iframe" not in html
    assert "tahmin yok" in html.lower()


def test_decode_heatmap_returns_image_when_present() -> None:
    img = decode_heatmap(_response(heatmap=_png_base64()))
    assert isinstance(img, Image.Image)
    assert img.size == (4, 4)


def test_decode_heatmap_none_when_absent_or_invalid() -> None:
    assert decode_heatmap(_response(heatmap=None)) is None
    assert decode_heatmap(_response(heatmap="not-base64!!")) is None


def test_confidence_label_high_vs_low() -> None:
    assert "Yüksek" in confidence_label(_response(is_confident=True))
    low = confidence_label(_response(is_confident=False))
    assert "DÜŞÜK" in low
