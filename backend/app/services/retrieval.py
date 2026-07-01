"""RetrievalService: FAISS top-K search + confidence / heatmap gating.

The index is a ``faiss.IndexIDMap`` over ``IndexFlatIP`` built offline by
``scripts/build_index.py`` (Sprint S1). Because descriptors are L2-normalized,
the inner product equals cosine similarity, so a raw ``IndexFlatIP`` score in
``[-1, 1]`` doubles as the confidence signal (see
docs/architecture/03-ai-pipeline.md and 04-api-sozlesmesi.md).

This module also owns the two pure decision helpers the API contract defines:
``is_confident`` (top-1 similarity vs threshold) and ``resolve_heatmap_status``
(why ``heatmap_png_base64`` is populated or null).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app.ml.aggregator import DESCRIPTOR_DIM
from app.schemas.common import HeatmapStatus


@dataclass(frozen=True)
class RetrievalHit:
    """A single FAISS match: database id, cosine similarity, 1-based rank."""

    db_id: int
    similarity: float
    rank: int


def resolve_heatmap_status(include_heatmap: bool, is_confident: bool) -> HeatmapStatus:
    """Decide the ``heatmap_status`` field value.

    Mirrors the API contract: the Eigen-CAM overlay is produced only when the
    caller requested it *and* the top-1 match is confident; otherwise it is
    skipped and the reason is reported.
    """
    if not include_heatmap:
        return "disabled"
    if not is_confident:
        return "skipped_low_confidence"
    return "included"


class RetrievalService:
    """Wrap a FAISS index for cosine top-K retrieval + confidence checks."""

    def __init__(self, index: faiss.Index, confidence_threshold: float = 0.5) -> None:
        """Wrap a prebuilt ``index`` and store the confidence threshold."""
        self._index = index
        self._threshold = confidence_threshold

    @classmethod
    def from_file(cls, path: Path, confidence_threshold: float = 0.5) -> RetrievalService:
        """Load a serialized FAISS index from ``path``."""
        index = faiss.read_index(str(path))
        return cls(index, confidence_threshold)

    @property
    def size(self) -> int:
        """Number of indexed database vectors."""
        return int(self._index.ntotal)

    def is_confident(self, similarity: float) -> bool:
        """True if a top-1 similarity clears the configured threshold."""
        return similarity >= self._threshold

    def search(self, query: np.ndarray, top_k: int) -> list[RetrievalHit]:
        """Return up to ``top_k`` matches for an L2-normalized query vector.

        Args:
            query: A ``(DESCRIPTOR_DIM,)`` or ``(1, DESCRIPTOR_DIM)`` float array.
            top_k: Maximum number of matches to return.

        Returns:
            Ranked ``RetrievalHit`` list (padding ``-1`` ids from FAISS dropped).
        """
        vec = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        if vec.shape[1] != DESCRIPTOR_DIM:
            raise ValueError(f"Query is {vec.shape[1]}-dim, expected {DESCRIPTOR_DIM}.")
        k = min(top_k, self.size)
        if k <= 0:
            return []
        sims, ids = self._index.search(vec, k)
        hits: list[RetrievalHit] = []
        for rank, (db_id, sim) in enumerate(zip(ids[0], sims[0], strict=True), start=1):
            if db_id == -1:
                continue
            hits.append(RetrievalHit(db_id=int(db_id), similarity=float(sim), rank=rank))
        return hits
