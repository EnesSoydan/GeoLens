"""XAIService tests: reshape/encoder (fast) + real-model plausibility (opt-in).

The corner-pattern test is gated behind ``GEOLENS_RUN_MODEL_SMOKE`` (like the
embedding smoke test) because it downloads/loads the real DINOv2+SALAD weights.
It does more than assert "no exception": it places a high-contrast patch in one
corner of an otherwise flat image and checks the Eigen-CAM saliency actually
concentrates its variation there (sign-invariant, since PCA sign is arbitrary).
"""

from __future__ import annotations

import base64
import io
import os

import numpy as np
import pytest
import torch
from PIL import Image

from app.services.xai import _vit_reshape_transform, encode_png_base64


def test_reshape_drops_single_cls_prefix():
    # DINOv2 ViT-B/14 @322: 530 tokens = 1 CLS + 529 patches (23x23).
    tokens = torch.randn(2, 530, 768)
    out = _vit_reshape_transform(tokens)
    assert out.shape == (2, 768, 23, 23)


def test_reshape_tolerates_register_tokens():
    # 534 tokens = 5 prefix (CLS + 4 registers) + 529 patches.
    tokens = torch.randn(1, 534, 768)
    out = _vit_reshape_transform(tokens)
    assert out.shape == (1, 768, 23, 23)


def test_reshape_perfect_square_has_no_prefix():
    tokens = torch.randn(1, 529, 768)
    out = _vit_reshape_transform(tokens)
    assert out.shape == (1, 768, 23, 23)


def test_encode_png_base64_roundtrip():
    arr = np.zeros((6, 4, 3), dtype=np.uint8)
    arr[1:3, 1:3] = 255
    encoded = encode_png_base64(arr)
    decoded = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert decoded.format == "PNG"
    assert decoded.size == (4, 6)  # PIL size is (width, height).
    assert np.array_equal(np.asarray(decoded.convert("RGB")), arr)


def _corner_pattern_image(size: int = 322, patch: int = 120) -> Image.Image:
    """Flat gray image with a high-contrast noise patch in the top-left corner."""
    rng = np.random.default_rng(0)
    canvas = np.full((size, size, 3), 128, dtype=np.uint8)
    canvas[:patch, :patch] = rng.integers(0, 256, size=(patch, patch, 3), dtype=np.uint8)
    return Image.fromarray(canvas)


@pytest.mark.skipif(
    not os.getenv("GEOLENS_RUN_MODEL_SMOKE"),
    reason="Requires real DINOv2+SALAD weights; set GEOLENS_RUN_MODEL_SMOKE=1.",
)
def test_eigen_cam_concentrates_on_salient_corner():
    from app.ml.backbone import load_vpr_model
    from app.services.xai import XAIService

    model = load_vpr_model(pretrained=True, device="cpu")
    xai = XAIService(model, image_size=322, device="cpu")
    image = _corner_pattern_image()

    mask = xai.heatmap_mask(image)
    assert mask.shape == (322, 322)
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0 + 1e-6

    # Sign-invariant plausibility: the patterned (top-left) quadrant should carry
    # more saliency variation than the flat (bottom-right) quadrant.
    q = 120
    top_left = mask[:q, :q]
    bottom_right = mask[-q:, -q:]
    tl_dev = float(np.mean(np.abs(top_left - mask.mean())))
    br_dev = float(np.mean(np.abs(bottom_right - mask.mean())))
    assert tl_dev > br_dev, f"saliency not concentrated on the pattern: {tl_dev=} {br_dev=}"

    # And the overlay encodes to a valid PNG of the expected size.
    decoded = Image.open(io.BytesIO(base64.b64decode(xai.heatmap_png_base64(image))))
    assert decoded.format == "PNG"
    assert decoded.size == (322, 322)
