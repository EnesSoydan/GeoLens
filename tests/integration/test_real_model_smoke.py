"""Opt-in smoke test that downloads the real DINOv2+SALAD weights.

Skipped by default (heavy ~350 MB torch.hub download). Enable with:

    GEOLENS_RUN_MODEL_SMOKE=1 pytest tests/integration/test_real_model_smoke.py
"""

import os

import numpy as np
import pytest

_ENABLED = os.getenv("GEOLENS_RUN_MODEL_SMOKE") == "1"
# Device for the single-device smoke test (default cpu; set to "cuda" to exercise
# the GPU path after a CUDA torch install).
_DEVICE = os.getenv("GEOLENS_SMOKE_DEVICE", "cpu")


@pytest.mark.skipif(not _ENABLED, reason="Set GEOLENS_RUN_MODEL_SMOKE=1 to run.")
def test_real_dinov2_salad_embed():
    from PIL import Image

    from app.services.embedding import EmbeddingService

    service = EmbeddingService.load(pretrained=True, device=_DEVICE)
    vec = service.embed(Image.new("RGB", (320, 240), (127, 127, 127)))

    assert vec.shape == (8448,)
    assert np.isfinite(vec).all()
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.skipif(not _ENABLED, reason="Set GEOLENS_RUN_MODEL_SMOKE=1 to run.")
def test_cpu_and_cuda_descriptors_agree():
    """Guard against a CUDA build silently producing different descriptors.

    A version/kernel mismatch can run without error yet return subtly wrong
    activations. Embedding the same image on CPU and CUDA must yield the same
    unit descriptor (cosine ~1.0) up to float rounding.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available.")

    from PIL import Image

    from app.services.embedding import EmbeddingService

    image = Image.new("RGB", (320, 240), (90, 140, 200))
    cpu_vec = EmbeddingService.load(pretrained=True, device="cpu").embed(image)
    cuda_vec = EmbeddingService.load(pretrained=True, device="cuda").embed(image)

    assert cuda_vec.shape == (8448,)
    assert np.isfinite(cuda_vec).all()
    assert np.linalg.norm(cuda_vec) == pytest.approx(1.0, abs=1e-4)
    cosine = float(np.dot(cpu_vec, cuda_vec))
    assert cosine == pytest.approx(1.0, abs=1e-3)
