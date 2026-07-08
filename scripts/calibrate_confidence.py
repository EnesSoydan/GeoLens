"""Calibrate ``confidence_threshold`` on the official MSLS-val set (Sprint S5).

``RetrievalService.is_confident`` flags a prediction as trustworthy when the
top-1 cosine similarity clears ``settings.confidence_threshold``. That flag drives
the UI's low-confidence warning and whether the Eigen-CAM heatmap is produced
(docs/architecture/04-api-sozlesmesi.md), so a good threshold should separate
**correct** top-1 retrievals (top-1 is a ≤25 m positive) from **incorrect** ones.

This script reuses the evaluation pipeline: it embeds the 750 ``subtask='all'``
MSLS-val queries, takes the top-1 retrieval, records ``(similarity, correct)`` per
query, then sweeps candidate thresholds. Because top-1 is correct ~92% of the
time, *accuracy* is a poor selection criterion (flagging everything confident
already scores ~0.92); we instead pick the threshold that maximises **Youden's J**
(``TPR - FPR``), the standard base-rate-robust operating point that best separates
the correct- and incorrect-top-1 similarity distributions. Precision (fraction of
*confident* predictions that are correct) and coverage (fraction flagged
confident) are reported at the recommendation, alongside a high-precision
(``precision >= 0.95``) reference point, so the trade-off is explicit.

Usage (the project root is on PYTHONPATH so ``scripts.evaluate`` helpers import)::

    PYTHONPATH=backend:. python scripts/calibrate_confidence.py --device cuda
        [--output data/index/confidence_calibration.json]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scripts.evaluate import (
    _embed_queries,
    _load_database,
    _load_queries,
    _read_allowed_query_keys,
    ground_truth_positive_ids,
)

from app.core.config import get_settings
from app.ml.faiss_io import read_index
from app.services.embedding import EmbeddingService


@dataclass(frozen=True)
class ConfidenceStats:
    """Confidence-flag quality at one threshold (flag = similarity >= threshold)."""

    threshold: float
    accuracy: float  # fraction where (confident == correct)
    precision: float  # of the confident predictions, fraction actually correct
    coverage: float  # fraction of queries flagged confident
    recall: float  # of the correct predictions, fraction flagged confident (TPR)
    f1: float
    youden_j: float  # TPR - FPR: separation quality, robust to class imbalance


def confidence_stats(sims: np.ndarray, correct: np.ndarray, threshold: float) -> ConfidenceStats:
    """Score how well ``similarity >= threshold`` predicts top-1 correctness."""
    confident = sims >= threshold
    tp = int(np.sum(confident & correct))
    fp = int(np.sum(confident & ~correct))
    fn = int(np.sum(~confident & correct))
    tn = int(np.sum(~confident & ~correct))
    n = len(sims)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return ConfidenceStats(
        threshold=float(threshold),
        accuracy=(tp + tn) / n if n else 0.0,
        precision=precision,
        coverage=(tp + fp) / n if n else 0.0,
        recall=recall,
        f1=f1,
        youden_j=recall - fpr,
    )


def _candidate_thresholds(sims: np.ndarray) -> np.ndarray:
    """Midpoints between consecutive observed similarities (+ extremes)."""
    ordered = np.unique(sims)
    if ordered.size == 1:
        return np.array([ordered[0] - 1e-6, ordered[0] + 1e-6])
    mids = (ordered[:-1] + ordered[1:]) / 2.0
    return np.concatenate(([ordered[0] - 1e-6], mids, [ordered[-1] + 1e-6]))


def recommend_threshold(
    sims: np.ndarray, correct: np.ndarray
) -> tuple[ConfidenceStats, list[ConfidenceStats]]:
    """Sweep candidate thresholds; return the max-Youden's-J point + the full curve.

    Youden's J (``TPR - FPR``) is used instead of accuracy because the ~92% base
    rate of correct top-1 retrievals makes accuracy degenerate (flag-everything
    already scores ~0.92). J rewards separating the correct/incorrect similarity
    distributions regardless of that skew. Ties are broken toward the higher (more
    conservative) threshold.
    """
    candidates = _candidate_thresholds(sims)
    curve = [confidence_stats(sims, correct, float(t)) for t in candidates]
    best = max(curve, key=lambda s: (s.youden_j, s.threshold))
    return best, curve


def high_precision_threshold(
    curve: list[ConfidenceStats], min_precision: float = 0.95
) -> ConfidenceStats | None:
    """Lowest-threshold point (max coverage) whose precision meets ``min_precision``."""
    eligible = [s for s in curve if s.precision >= min_precision]
    return min(eligible, key=lambda s: s.threshold) if eligible else None


def main() -> int:
    """Embed val queries, record top-1 (similarity, correct), recommend a threshold."""
    parser = argparse.ArgumentParser(description="Calibrate confidence_threshold on MSLS-val.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threshold", type=float, default=25.0, help="GT distance (m).")
    parser.add_argument("--subtask", default="all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--city", action="append")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    raw_dir = settings.msls_raw_dir or (settings.raw_dir / "msls_images")
    index_path = settings.index_dir / "index.faiss"
    db_path = settings.index_dir / "metadata.db"
    for path in (index_path, db_path):
        if not path.exists():
            print(f"error: missing artifact {path} (run build_index.py first)", file=sys.stderr)
            return 2
    cities = tuple(args.city) if args.city else settings.target_cities

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    db_ids, db_lat, db_lon = _load_database(conn)
    allowed = {c: _read_allowed_query_keys(raw_dir, c, args.subtask) for c in cities}
    queries = _load_queries(conn, cities, allowed)
    conn.close()
    if not queries:
        print("error: no queries selected", file=sys.stderr)
        return 2

    print(f"queries={len(queries)} database={len(db_ids)} device={args.device}", flush=True)
    embedding = EmbeddingService.load(pretrained=True, device=args.device)
    index = read_index(index_path)
    query_vectors = _embed_queries(embedding, queries, args.batch_size)

    sims_all, ids_all = index.search(query_vectors, 1)  # top-1 similarity + id
    sims: list[float] = []
    correct: list[bool] = []
    for qi, query in enumerate(queries):
        positives = ground_truth_positive_ids(
            query.lat, query.lon, db_ids, db_lat, db_lon, args.threshold
        )
        if not positives:
            continue  # unlocalizable query (no positive in DB) -> excluded, as in eval
        sims.append(float(sims_all[qi, 0]))
        correct.append(int(ids_all[qi, 0]) in positives)
    sims_arr = np.array(sims, dtype=np.float64)
    correct_arr = np.array(correct, dtype=bool)

    best, curve = recommend_threshold(sims_arr, correct_arr)
    hi_prec = high_precision_threshold(curve, min_precision=0.95)
    grid = [confidence_stats(sims_arr, correct_arr, t) for t in np.arange(0.30, 0.66, 0.05)]

    def _point(s: ConfidenceStats) -> dict[str, float]:
        return {
            "threshold": round(s.threshold, 4),
            "accuracy": round(s.accuracy, 4),
            "precision": round(s.precision, 4),
            "coverage": round(s.coverage, 4),
            "recall": round(s.recall, 4),
            "youden_j": round(s.youden_j, 4),
        }

    report = {
        "n_queries": int(correct_arr.size),
        "n_correct_top1": int(np.sum(correct_arr)),
        "base_rate_correct": round(float(np.mean(correct_arr)), 4),
        "sim_correct_mean": round(float(sims_arr[correct_arr].mean()), 4),
        "sim_incorrect_mean": round(float(sims_arr[~correct_arr].mean()), 4)
        if np.any(~correct_arr)
        else None,
        "recommended_threshold": round(best.threshold, 4),
        "recommendation_criterion": "max Youden's J",
        "at_recommended": _point(best),
        "high_precision_point": _point(hi_prec) if hi_prec is not None else None,
        "grid_0.30_to_0.65_step_0.05": [_point(s) for s in grid],
    }
    print("\n=== confidence calibration ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
