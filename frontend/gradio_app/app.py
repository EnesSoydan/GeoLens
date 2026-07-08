"""Gradio Blocks UI - upload, top-K list, folium map, heatmap (Sprint S4).

The UI is a thin shell over the pure helpers in :mod:`rendering` and the HTTP
:mod:`api_client`: the button callback turns the uploaded image + controls into a
``POST /predict`` call and fans the response out to the map, heatmap, table and
confidence widgets. It never touches the model directly (docs/architecture/
05-frontend.md). ``build_ui`` returns the Blocks app without launching so it can
be mounted on FastAPI (``/ui``) or launched standalone.
"""

from __future__ import annotations

import io
from typing import Any

import gradio as gr
from PIL import Image

from frontend.gradio_app.api_client import GeoLensClient, PredictionError
from frontend.gradio_app.rendering import (
    TABLE_HEADERS,
    build_map_html,
    confidence_label,
    decode_heatmap,
    predictions_to_rows,
)

_EMPTY_MAP = "<p style='padding:1rem;color:#666'>Sonuç bekleniyor…</p>"


def _image_to_png_bytes(image: Image.Image) -> bytes:
    """Encode a PIL image to PNG bytes for the multipart upload."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _run_prediction(
    client: GeoLensClient,
    image: Image.Image | None,
    top_k: int,
    include_heatmap: bool,
) -> tuple[str, str, Image.Image | None, list[list[Any]]]:
    """Callback: image + controls -> (confidence, map HTML, heatmap, table rows).

    Returns an error message (and empty map/table) instead of raising so the UI
    surfaces backend failures gracefully.
    """
    if image is None:
        return "Lütfen bir sorgu görüntüsü yükleyin.", _EMPTY_MAP, None, []
    try:
        response = client.predict(
            image_bytes=_image_to_png_bytes(image),
            filename="query.png",
            content_type="image/png",
            top_k=int(top_k),
            include_heatmap=bool(include_heatmap),
        )
    except PredictionError as exc:
        return str(exc), _EMPTY_MAP, None, []

    predictions = response["predictions"]
    return (
        confidence_label(response),
        build_map_html(predictions),
        decode_heatmap(response),
        predictions_to_rows(predictions),
    )


def build_ui(client: GeoLensClient | None = None) -> gr.Blocks:
    """Build (but do not launch) the GeoLens Gradio Blocks app."""
    client = client or GeoLensClient()

    with gr.Blocks(title="GeoLens") as demo:
        gr.Markdown(
            "# GeoLens\n"
            "Bir sokak görüntüsü yükleyin; model en benzer referansları bulup "
            "konumu tahmin eder ve modelin nereye baktığını Eigen-CAM ile gösterir."
        )
        gr.Markdown(
            "> **En iyi sonuç için gerçek sokak-seviyesi fotoğraflar kullanın.** "
            "Referans indeksi yalnızca **Kopenhag** ve **San Francisco**'nun MSLS "
            "sokak görüntülerinden oluşur. Turistik/rastgele web fotoğrafları, "
            "iç mekân veya bu iki şehir dışındaki görüntüler düşük güven ve yanlış "
            "tahmin verebilir — bu, modelin dağılım-dışı (out-of-distribution) "
            "girdideki beklenen davranışıdır."
        )
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Sorgu görüntüsü")
                top_k = gr.Slider(1, 10, value=5, step=1, label="top_k")
                include_heatmap = gr.Checkbox(value=True, label="Isı haritası (Eigen-CAM)")
                predict_btn = gr.Button("Tahmin Et", variant="primary")
            with gr.Column():
                confidence_out = gr.Textbox(label="Güven", interactive=False)
                map_out = gr.HTML(label="Harita", value=_EMPTY_MAP)
                heatmap_out = gr.Image(label="Eigen-CAM overlay", type="pil")
                table_out = gr.Dataframe(
                    headers=TABLE_HEADERS, label="Tahminler", interactive=False
                )

        predict_btn.click(
            fn=lambda img, k, hm: _run_prediction(client, img, k, hm),
            inputs=[image_in, top_k, include_heatmap],
            outputs=[confidence_out, map_out, heatmap_out, table_out],
        )
    return demo


def main() -> None:
    """Launch the standalone Gradio app (backend must be running separately)."""
    build_ui().launch()


if __name__ == "__main__":
    main()
