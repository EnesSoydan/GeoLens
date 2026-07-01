"""EmbeddingService logic tests using a fake model + synthetic images.

These tests validate preprocessing, batching, L2-normalization and the
dimension guard without downloading the real (heavy) VPR weights.
"""

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from app.ml.aggregator import DESCRIPTOR_DIM
from app.services.embedding import EmbeddingService


class _FakeModel(nn.Module):
    """Returns deterministic-shaped random descriptors of ``dim`` size."""

    def __init__(self, dim: int = DESCRIPTOR_DIM) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Non-normalized on purpose, to prove the service normalizes.
        return torch.randn(x.shape[0], self.dim) * 5.0


def _synthetic_image(w: int = 320, h: int = 240) -> Image.Image:
    rng = np.random.default_rng(0)
    arr = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def service() -> EmbeddingService:
    return EmbeddingService(_FakeModel(), device="cpu")


def test_preprocess_shape(service):
    tensor = service.preprocess(_synthetic_image())
    assert tuple(tensor.shape) == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_embed_shape_and_norm(service):
    vec = service.embed(_synthetic_image())
    assert vec.shape == (DESCRIPTOR_DIM,)
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all()
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)


def test_embed_batch(service):
    imgs = [_synthetic_image(), _synthetic_image(200, 200), _synthetic_image()]
    out = service.embed_batch(imgs)
    assert out.shape == (3, DESCRIPTOR_DIM)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_wrong_dim_model_raises():
    svc = EmbeddingService(_FakeModel(dim=512), device="cpu")
    with pytest.raises(ValueError, match="expected 8448"):
        svc.embed(_synthetic_image())
