"""EmbeddingService - image to L2-normalized DINOv2+SALAD descriptor.

The service owns the preprocessing transform (resize 224x224 + ImageNet
normalize) and the VPR model, exposing single- and batch-image embedding. The
model is injected for testability; ``EmbeddingService.load`` is the production
factory that fetches the pretrained weights.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from app.ml.aggregator import DESCRIPTOR_DIM
from app.ml.model_registry import IMAGENET_MEAN, IMAGENET_STD


class EmbeddingService:
    """Produce L2-normalized global descriptors from images."""

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        image_size: int = 322,
    ) -> None:
        """Wrap a VPR ``model`` with its preprocessing transform."""
        self._model = model
        self._device = torch.device(device)
        self._dtype = dtype
        self._transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    @classmethod
    def load(
        cls,
        pretrained: bool = True,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        image_size: int = 322,
    ) -> EmbeddingService:
        """Load the pretrained VPR model and wrap it in a service."""
        # Lazy import keeps the torch.hub download out of module import time.
        from app.ml.backbone import load_vpr_model

        model = load_vpr_model(pretrained=pretrained, device=device, dtype=dtype)
        return cls(model, device=device, dtype=dtype, image_size=image_size)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Convert a PIL image to a normalized CHW float tensor."""
        tensor: torch.Tensor = self._transform(image.convert("RGB"))
        return tensor

    def embed(self, image: Image.Image) -> np.ndarray:
        """Return the L2-normalized descriptor for a single image."""
        return self.embed_batch([image])[0]

    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Return L2-normalized descriptors for a batch of images.

        Returns:
            A ``(len(images), DESCRIPTOR_DIM)`` float32 array of unit vectors.
        """
        batch = torch.stack([self.preprocess(img) for img in images])
        batch = batch.to(device=self._device, dtype=self._dtype)
        with torch.inference_mode():
            descriptors = self._model(batch)
        normalized = torch.nn.functional.normalize(descriptors.float(), p=2, dim=1)
        out: np.ndarray = normalized.cpu().numpy().astype(np.float32)
        if out.shape[1] != DESCRIPTOR_DIM:
            raise ValueError(f"Model returned {out.shape[1]}-dim, expected {DESCRIPTOR_DIM}.")
        return out
