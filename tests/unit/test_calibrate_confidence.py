"""Unit tests for the pure confidence-calibration helpers (no data/GPU) — Sprint S5."""

from __future__ import annotations

import numpy as np
import pytest
from scripts.calibrate_confidence import (
    confidence_stats,
    high_precision_threshold,
    recommend_threshold,
)


def test_confidence_stats_perfect_separation() -> None:
    sims = np.array([0.9, 0.8, 0.2, 0.1])
    correct = np.array([True, True, False, False])
    stats = confidence_stats(sims, correct, threshold=0.5)
    assert stats.accuracy == pytest.approx(1.0)
    assert stats.precision == pytest.approx(1.0)
    assert stats.coverage == pytest.approx(0.5)
    assert stats.recall == pytest.approx(1.0)
    assert stats.f1 == pytest.approx(1.0)
    assert stats.youden_j == pytest.approx(1.0)  # TPR 1 - FPR 0


def test_confidence_stats_threshold_too_low_flags_everything() -> None:
    sims = np.array([0.9, 0.2])
    correct = np.array([True, False])
    stats = confidence_stats(sims, correct, threshold=0.0)
    assert stats.coverage == pytest.approx(1.0)  # both flagged confident
    assert stats.precision == pytest.approx(0.5)  # one of two confident is correct


def test_confidence_stats_no_confident_predictions() -> None:
    sims = np.array([0.4, 0.3])
    correct = np.array([True, False])
    stats = confidence_stats(sims, correct, threshold=0.9)
    assert stats.coverage == pytest.approx(0.0)
    assert stats.precision == pytest.approx(0.0)  # undefined -> 0 (no confident preds)


def test_recommend_threshold_finds_separating_point() -> None:
    sims = np.array([0.95, 0.90, 0.30, 0.25])
    correct = np.array([True, True, False, False])
    best, curve = recommend_threshold(sims, correct)
    # The best split lies strictly between 0.30 (incorrect) and 0.90 (correct).
    assert 0.30 < best.threshold < 0.90
    assert best.youden_j == pytest.approx(1.0)
    assert best.accuracy == pytest.approx(1.0)
    assert len(curve) >= 2


def test_recommend_ignores_base_rate_skew() -> None:
    # Heavy positive skew: max-accuracy would flag everything; Youden's J instead
    # picks a threshold that separates the lone incorrect (low-sim) case.
    sims = np.array([0.9, 0.85, 0.82, 0.8, 0.3])
    correct = np.array([True, True, True, True, False])
    best, _ = recommend_threshold(sims, correct)
    assert best.threshold > 0.3  # does not simply flag all as confident
    assert best.youden_j == pytest.approx(1.0)


def test_high_precision_threshold_picks_lowest_meeting_target() -> None:
    sims = np.array([0.9, 0.8, 0.6, 0.55, 0.2])
    correct = np.array([True, True, True, False, False])
    _, curve = recommend_threshold(sims, correct)
    point = high_precision_threshold(curve, min_precision=1.0)
    assert point is not None
    # Must exclude the incorrect 0.55 -> threshold sits between 0.55 and 0.60.
    assert 0.55 < point.threshold <= 0.60
    assert point.precision == pytest.approx(1.0)


def test_high_precision_threshold_none_when_unreachable() -> None:
    sims = np.array([0.9, 0.9])
    correct = np.array([True, False])  # every confident set is 50% precise at best
    _, curve = recommend_threshold(sims, correct)
    assert high_precision_threshold(curve, min_precision=0.95) is None
