"""DINOv2 ViT-B/14 + SALAD VPR model loading.

The pretrained VPR model bundles the DINOv2 ViT-B/14 backbone with the SALAD
optimal-transport aggregation head (Izquierdo & Civera, CVPR 2024) into a single
``nn.Module`` that maps an image batch to 8448-dim global descriptors. Weights
are trained on GSV-Cities and fetched via ``torch.hub`` from the official SALAD
repository (see docs/architecture/03-ai-pipeline.md).
"""

from __future__ import annotations

import torch
from torch import nn

# Official SALAD repo exposes ``dinov2_salad`` through its ``hubconf.py``.
DINOV2_SALAD_HUB = "serizba/salad"
DINOV2_SALAD_ENTRYPOINT = "dinov2_salad"


def load_vpr_model(
    pretrained: bool = True,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Load the DINOv2+SALAD model and put it in eval mode.

    Args:
        pretrained: Load the GSV-Cities pretrained weights via ``torch.hub``.
        device: Target device (``"cpu"`` or ``"cuda"``).
        dtype: Parameter dtype. fp16 is only meaningful on CUDA; on the RTX 4060
            it keeps the 8GB VRAM budget (see docs/architecture/03-ai-pipeline.md).

    Returns:
        The VPR model on ``device`` in eval mode.
    """
    # ``trust_repo=True`` skips the interactive trust prompt for the third-party
    # SALAD repo (required in non-interactive / server contexts).
    model: nn.Module = torch.hub.load(
        DINOV2_SALAD_HUB,
        DINOV2_SALAD_ENTRYPOINT,
        pretrained=pretrained,
        trust_repo=True,
    )
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model
