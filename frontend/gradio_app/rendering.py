"""Pure rendering helpers: response dict -> map HTML, table rows, heatmap, label.

These functions are deliberately free of Gradio and network I/O so they can be
unit-tested against a fixed ``PredictionResponse`` fixture (Sprint S4 step 1,
data/GPU-independent). The Gradio callback in ``app.py`` only wires them to
widgets.

Map behaviour follows docs/architecture/05-frontend.md: top-K markers, the top-1
match highlighted, and the view fit to all markers. On low confidence
(``is_confident == false``) the backend already withholds the heatmap; the UI
surfaces a warning via :func:`confidence_label`.
"""

from __future__ import annotations

import base64
import binascii
import io
from typing import Any

import folium
from PIL import Image, UnidentifiedImageError

TABLE_HEADERS = ["Sıra", "Enlem", "Boylam", "Benzerlik", "Referans ID"]


def build_map_html(predictions: list[dict[str, Any]]) -> str:
    """Render a folium (Leaflet) map of the ranked predictions as an HTML iframe.

    The top-1 marker is red and larger; the rest are blue. The map view is fit to
    the bounds of all markers. Returns a placeholder message when there are no
    predictions (e.g. after an error).
    """
    if not predictions:
        return "<p style='padding:1rem;color:#666'>Harita için tahmin yok.</p>"

    lats = [float(p["lat"]) for p in predictions]
    lons = [float(p["lon"]) for p in predictions]
    fmap = folium.Map(location=[lats[0], lons[0]], zoom_start=13, tiles="OpenStreetMap")

    for pred in predictions:
        is_top = int(pred["rank"]) == 1
        popup = (
            f"#{pred['rank']} · benzerlik {float(pred['similarity']):.3f}<br>"
            f"{pred.get('reference_image_id', '')}"
        )
        folium.Marker(
            location=[float(pred["lat"]), float(pred["lon"])],
            popup=folium.Popup(popup, max_width=250),
            icon=folium.Icon(
                color="red" if is_top else "blue",
                icon="star" if is_top else "info-sign",
            ),
        ).add_to(fmap)

    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    html: str = fmap._repr_html_()
    return html


def predictions_to_rows(predictions: list[dict[str, Any]]) -> list[list[Any]]:
    """Convert predictions to ``gr.Dataframe`` rows matching ``TABLE_HEADERS``."""
    return [
        [
            int(p["rank"]),
            round(float(p["lat"]), 5),
            round(float(p["lon"]), 5),
            round(float(p["similarity"]), 4),
            p.get("reference_image_id", ""),
        ]
        for p in predictions
    ]


def decode_heatmap(response: dict[str, Any]) -> Image.Image | None:
    """Decode the base64 PNG Eigen-CAM overlay to a PIL image, or ``None``.

    Returns ``None`` when the backend produced no overlay (``heatmap_png_base64``
    is null, e.g. status ``disabled`` / ``skipped_low_confidence``) or the payload
    is not decodable.
    """
    encoded = response.get("heatmap_png_base64")
    if not encoded:
        return None
    try:
        raw = base64.b64decode(encoded, validate=True)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (binascii.Error, ValueError, UnidentifiedImageError):
        return None


def confidence_label(response: dict[str, Any]) -> str:
    """Build the confidence line, warning the user when the top-1 is low-confidence."""
    confidence = float(response["confidence"])
    if response.get("is_confident"):
        return f"Güven: {confidence:.3f} — Yüksek"
    return (
        f"Güven: {confidence:.3f} — DÜŞÜK. Tahmin ve harita güvenilmez olabilir; "
        "ısı haritası bu nedenle gösterilmez."
    )
