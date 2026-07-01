"""SALAD aggregation head descriptor specification.

SALAD aggregates DINOv2 patch tokens via optimal transport into a fixed set of
clusters, so the global descriptor dimension is independent of input resolution:

    64 clusters x 128 dims  = 8192   (local aggregation)
    + global token          =  256
    --------------------------------
    total                   = 8448

The pretrained head itself ships inside the ``dinov2_salad`` hub model
(see ``backbone.load_vpr_model``); this module only pins the descriptor contract.
"""

from __future__ import annotations

NUM_CLUSTERS = 64
CLUSTER_DIM = 128
GLOBAL_DIM = 256
DESCRIPTOR_DIM = NUM_CLUSTERS * CLUSTER_DIM + GLOBAL_DIM  # 8448


def check_descriptor_dim(length: int) -> None:
    """Raise ``ValueError`` if ``length`` is not the SALAD descriptor dim."""
    if length != DESCRIPTOR_DIM:
        raise ValueError(f"Expected {DESCRIPTOR_DIM}-dim descriptor, got {length}.")
