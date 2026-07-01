"""Opt-in smoke test that downloads the real DINOv2+SALAD weights.

Skipped by default (heavy ~350 MB torch.hub download). Enable with:

    GEOLENS_RUN_MODEL_SMOKE=1 pytest tests/integration/test_real_model_smoke.py
"""

import os

import numpy as np
import pytest

_ENABLED = os.getenv("GEOLENS_RUN_MODEL_SMOKE") == "1"


@pytest.mark.skipif(not _ENABLED, reason="Set GEOLENS_RUN_MODEL_SMOKE=1 to run.")
def test_real_dinov2_salad_embed():
    from PIL import Image

    from app.services.embedding import EmbeddingService

    service = EmbeddingService.load(pretrained=True, device="cpu")
    vec = service.embed(Image.new("RGB", (320, 240), (127, 127, 127)))

    assert vec.shape == (8448,)
    assert np.isfinite(vec).all()
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-4)
