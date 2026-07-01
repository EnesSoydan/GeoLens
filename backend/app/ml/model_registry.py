"""Model manifest / version registry.

The manifest captures everything needed to reproduce an index build and to
detect model/index drift at startup: a hash/identity mismatch between the loaded
model and the index manifest raises ``ModelNotReadyError`` (see
docs/architecture/03-ai-pipeline.md). It is stored as a JSON sidecar next to the
FAISS index.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.ml.aggregator import DESCRIPTOR_DIM

# ImageNet statistics used by the preprocessing transform.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Fields that define model+preprocessing identity (index_size is excluded).
_IDENTITY_FIELDS = (
    "model_name",
    "hub_source",
    "embedding_dim",
    "image_size",
    "mean",
    "std",
)


@dataclass(frozen=True)
class ModelManifest:
    """Reproducibility manifest for a built FAISS index."""

    model_name: str = "dinov2_salad"
    hub_source: str = "serizba/salad"
    embedding_dim: int = DESCRIPTOR_DIM
    image_size: int = 224
    mean: tuple[float, ...] = IMAGENET_MEAN
    std: tuple[float, ...] = IMAGENET_STD
    dataset: str = "msls_amsterdam"
    index_size: int = 0
    weights_sha256: str | None = None

    def to_json(self, path: Path) -> None:
        """Write the manifest to ``path`` as pretty JSON."""
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> ModelManifest:
        """Load a manifest from a JSON sidecar."""
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        data["mean"] = tuple(data["mean"])
        data["std"] = tuple(data["std"])
        return cls(**data)

    def is_compatible(self, other: ModelManifest) -> bool:
        """True if model identity + preprocessing match (ignores ``index_size``)."""
        return all(getattr(self, f) == getattr(other, f) for f in _IDENTITY_FIELDS)


def default_manifest(index_size: int = 0, weights_sha256: str | None = None) -> ModelManifest:
    """Return the canonical manifest for the MVP model."""
    return ModelManifest(index_size=index_size, weights_sha256=weights_sha256)
