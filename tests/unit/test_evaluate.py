"""Unit tests for the pure MSLS-val metric helpers (no data/GPU) — Sprint S5."""

from __future__ import annotations

import numpy as np
import pytest
from scripts.evaluate import (
    ground_truth_positive_ids,
    haversine_km,
    query_hit_flags,
)


def test_haversine_zero_distance() -> None:
    assert float(haversine_km(55.0, 12.0, 55.0, 12.0)) == pytest.approx(0.0, abs=1e-9)


def test_haversine_known_one_degree_latitude() -> None:
    # One degree of latitude is ~111.19 km along a great circle.
    assert float(haversine_km(0.0, 0.0, 1.0, 0.0)) == pytest.approx(111.19, abs=0.2)


def test_haversine_broadcasts_over_arrays() -> None:
    lat = np.array([55.0, 55.001])
    lon = np.array([12.0, 12.0])
    out = haversine_km(55.0, 12.0, lat, lon)
    assert out.shape == (2,)
    assert out[0] == pytest.approx(0.0, abs=1e-9)
    assert out[1] > 0.0


def test_ground_truth_selects_only_within_threshold() -> None:
    db_ids = np.array([10, 20, 30], dtype=np.int64)
    db_lat = np.array([55.0000, 55.0001, 55.0100])  # ~0 m, ~11 m, ~1.1 km
    db_lon = np.array([12.0000, 12.0000, 12.0000])
    positives = ground_truth_positive_ids(55.0, 12.0, db_ids, db_lat, db_lon, threshold_m=25.0)
    assert positives == {10, 20}


def test_ground_truth_empty_when_all_far() -> None:
    db_ids = np.array([1], dtype=np.int64)
    positives = ground_truth_positive_ids(
        55.0, 12.0, db_ids, np.array([56.0]), np.array([13.0]), threshold_m=25.0
    )
    assert positives == set()


def test_query_hit_flags_respects_rank_cutoffs() -> None:
    ranked = [5, 9, 3, 7, 1]  # positive (3) first appears at rank 3
    flags = query_hit_flags(ranked, positives={3}, ks=(1, 5, 10))
    assert flags == [False, True, True]


def test_query_hit_flags_top1_hit() -> None:
    flags = query_hit_flags([42, 7], positives={42}, ks=(1, 5))
    assert flags == [True, True]


def test_query_hit_flags_no_hit() -> None:
    flags = query_hit_flags([1, 2, 3], positives={99}, ks=(1, 3))
    assert flags == [False, False]
