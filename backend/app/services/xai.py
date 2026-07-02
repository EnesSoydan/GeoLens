"""XAIService: class-agnostic Eigen-CAM heatmap for the retrieval model.

Retrieval has no class logit, so gradient/class-based CAMs do not apply. Eigen-CAM
explains *where the model looked* purely from the target layer's activations: it
projects them onto their first principal component (no target, no backprop), which
matches the Faz C decision (docs/research/faz-c.md) and the API contract's
confidence gate (only produced when the top-1 match is confident).

Target layer + reshape were verified against the loaded ``serizba/salad`` model,
not assumed:

* structure ``model.backbone.model.blocks[-1].norm1`` (DINOv2 ViT-B/14) — the
  standard pytorch-grad-cam ViT target layer;
* at 322x322 its output is ``(B, 530, 768)`` = 1 CLS prefix + 529 patch tokens
  (23x23 grid, since 322/14 = 23). ``_vit_reshape_transform`` derives the grid
  from the largest square <= token count and drops the leading prefix tokens, so
  it also tolerates DINOv2 variants that add register tokens.
"""

from __future__ import annotations

import base64
import io
import math
from typing import Any, Protocol, cast

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import nn
from torchvision import transforms

from app.ml.model_registry import IMAGENET_MEAN, IMAGENET_STD


class HeatmapGenerator(Protocol):
    """Produce a base64-encoded PNG heatmap overlay for a query image."""

    def heatmap_png_base64(self, image: Image.Image) -> str:
        """Return the Eigen-CAM overlay as a base64 PNG string."""
        ...


def _resolve_target_layer(model: nn.Module) -> nn.Module:
    """Locate the DINOv2 last-block norm used as the Eigen-CAM target layer."""
    module: Any = model
    try:
        layer = module.backbone.model.blocks[-1].norm1
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError(
            "Unexpected model structure: cannot locate the ViT target layer "
            "'backbone.model.blocks[-1].norm1'."
        ) from exc
    return cast(nn.Module, layer)


def _vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape ViT token activations ``(B, N, C)`` to a spatial map ``(B, C, H, W)``.

    ``N`` is ``num_prefix + grid*grid``. The grid side is the largest integer whose
    square does not exceed ``N``; the leading ``N - grid*grid`` prefix tokens (CLS,
    and register tokens if present) are dropped before reshaping.
    """
    n_tokens = tensor.size(1)
    grid = math.isqrt(n_tokens)
    patches = tensor[:, n_tokens - grid * grid :, :]
    b, _, c = patches.shape
    reshaped = patches.reshape(b, grid, grid, c)
    return reshaped.permute(0, 3, 1, 2)


def encode_png_base64(rgb_uint8: np.ndarray) -> str:
    """Encode an ``(H, W, 3)`` uint8 RGB array as a base64 PNG string."""
    buffer = io.BytesIO()
    Image.fromarray(rgb_uint8).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class XAIService:
    """Wrap Eigen-CAM over the VPR model to produce heatmap overlays."""

    def __init__(
        self,
        model: nn.Module,
        image_size: int = 322,
        device: str | torch.device = "cpu",
    ) -> None:
        """Build the Eigen-CAM engine and the preprocessing transform."""
        self._image_size = image_size
        self._device = torch.device(device)
        self._transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self._cam = EigenCAM(
            model=model,
            target_layers=[_resolve_target_layer(model)],
            reshape_transform=_vit_reshape_transform,
        )

    def heatmap_mask(self, image: Image.Image) -> np.ndarray:
        """Return the ``(image_size, image_size)`` Eigen-CAM saliency in ``[0, 1]``."""
        rgb = image.convert("RGB")
        input_tensor = self._transform(rgb).unsqueeze(0).to(self._device)
        # EigenCAM ignores ``targets`` (class-agnostic); output is scaled to input.
        grayscale = self._cam(input_tensor=input_tensor, targets=None)
        mask: np.ndarray = np.asarray(grayscale[0], dtype=np.float32)
        return mask

    def heatmap_png_base64(self, image: Image.Image) -> str:
        """Return a jet-colormap Eigen-CAM overlay on the query as a base64 PNG.

        The saliency is rendered as a JET heatmap and alpha-blended *over the
        original query image* (not a bare saliency map), matching the API
        contract's ``overlay`` semantics (docs/architecture/04-api-sozlesmesi.md).
        ``image_weight=0.6`` keeps the original scene recognizable (heatmap ~40%).
        """
        rgb = image.convert("RGB").resize((self._image_size, self._image_size))
        rgb_float = np.asarray(rgb, dtype=np.float32) / 255.0
        overlay = show_cam_on_image(
            rgb_float, self.heatmap_mask(image), use_rgb=True, image_weight=0.6
        )
        return encode_png_base64(overlay)
