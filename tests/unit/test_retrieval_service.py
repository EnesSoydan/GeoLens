"""RetrievalService unit tests over a synthetic in-memory FAISS index.

The synthetic index (a handful of normalized 8448-dim vectors with custom ids)
validates the retrieval logic only. It does NOT stand in for the real
role-split, leakage-safe MSLS index built in Sprint S1.
"""

from __future__ import annotations

import faiss
import numpy as np
import pytest

from app.ml.aggregator import DESCRIPTOR_DIM
from app.schemas.common import HeatmapStatus
from app.services.retrieval import RetrievalService, resolve_heatmap_status

_N = 8
_DB_IDS = np.arange(100, 100 + _N, dtype=np.int64)


def _unit(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


@pytest.fixture
def synthetic_service() -> tuple[RetrievalService, np.ndarray]:
    """Build an IndexIDMap(IndexFlatIP) over N normalized synthetic vectors."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((_N, DESCRIPTOR_DIM)).astype(np.float32)
    vectors = np.ascontiguousarray([_unit(v) for v in vectors], dtype=np.float32)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(DESCRIPTOR_DIM))
    index.add_with_ids(vectors, _DB_IDS)
    return RetrievalService(index, confidence_threshold=0.5), vectors


def test_size_reports_vector_count(synthetic_service):
    service, _ = synthetic_service
    assert service.size == _N


def test_search_returns_self_as_top_hit(synthetic_service):
    service, vectors = synthetic_service
    hits = service.search(vectors[3], top_k=5)
    assert len(hits) == 5
    assert hits[0].db_id == int(_DB_IDS[3])
    assert hits[0].rank == 1
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-4)
    # Ranks are contiguous and 1-based; similarities are non-increasing.
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    sims = [h.similarity for h in hits]
    assert sims == sorted(sims, reverse=True)


def test_search_clamps_top_k_to_index_size(synthetic_service):
    service, vectors = synthetic_service
    hits = service.search(vectors[0], top_k=100)
    assert len(hits) == _N


def test_search_rejects_wrong_dimension(synthetic_service):
    service, _ = synthetic_service
    with pytest.raises(ValueError, match="expected"):
        service.search(np.zeros(16, dtype=np.float32), top_k=1)


def test_is_confident_threshold(synthetic_service):
    service, _ = synthetic_service
    assert service.is_confident(0.5) is True
    assert service.is_confident(0.51) is True
    assert service.is_confident(0.49) is False


@pytest.mark.parametrize(
    ("include_heatmap", "is_confident", "expected"),
    [
        (False, True, "disabled"),
        (False, False, "disabled"),
        (True, False, "skipped_low_confidence"),
        (True, True, "included"),
    ],
)
def test_resolve_heatmap_status(include_heatmap, is_confident, expected):
    result: HeatmapStatus = resolve_heatmap_status(include_heatmap, is_confident)
    assert result == expected
