"""Offline evaluation on the official mapillary_sls MSLS-val protocol (Sprint S5).

Reproduces the reference protocol used by the MSLS benchmark and the SALAD paper
so our Recall@N is directly comparable to the literature:

* **Split:** the official validation cities ``cph`` + ``sf`` (Copenhagen + San
  Francisco) — ``default_cities['val']`` in mapillary_sls.
* **Queries:** the ``subtask='all'`` subset selected by ``query/subtask_index.csv``
  (``all == True``): 502 (cph) + 248 (sf) = **750** queries. The remaining query
  images are not part of the official im2im validation and are excluded.
* **Database:** the full indexed reference set. For ``subtask='all'`` every
  database image is flagged (``database/subtask_index.csv`` all-True), so the
  FAISS index built by ``build_index.py`` already *is* the official val database.
* **Ground truth:** a database image is a positive for a query iff their GPS
  positions are within **25 m** — distance only. The reference
  ``mapillary_sls/evaluate.py`` uses ``--threshold 25`` and defines positives via
  ``NearestNeighbors.radius_neighbors`` on UTM coordinates with **no orientation
  term**; ``mapillary_sls/utils/eval.py::recall`` then matches the top-k ranking
  against exactly those distance-only positives. The ``ca`` (compass heading)
  column exists in ``raw.csv`` but the official protocol does *not* use it, so we
  intentionally do not apply a viewpoint/heading filter (see
  docs/architecture/08-degerlendirme.md).
* **Metrics:** Recall@{1,5,10,20} (a query counts as correct@k if any of its
  top-k retrievals is a positive) plus a median/mean km localisation error of the
  top-1 retrieval (our own diagnostic, not part of the official metric).

Queries with no positive within the threshold are dropped from the recall
denominator, mirroring the reference implementation (its ``qIdx`` keeps only
queries that have at least one positive).

Data-leakage: query rows never enter the FAISS index (``build_index.py`` embeds
only ``image_role='database'``), so a query cannot retrieve itself.

Usage::

    PYTHONPATH=backend python scripts/evaluate.py [--device cpu|cuda]
        [--threshold 25] [--subtask all] [--limit N] [--output metrics.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from PIL import Image

# scripts/ is run with PYTHONPATH=backend so ``app`` is importable.
from app.core.config import get_settings
from app.ml.faiss_io import read_index
from app.services.embedding import EmbeddingService

EARTH_RADIUS_KM = 6371.0088
DEFAULT_KS: tuple[int, ...] = (1, 5, 10, 20)
GT_THRESHOLD_M = 25.0


@dataclass(frozen=True)
class QueryRow:
    """One MSLS-val query: GPS truth + image path for embedding."""

    key: str
    image_path: str
    lat: float
    lon: float
    city: str


def haversine_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> np.ndarray:
    """Great-circle distance in kilometres. Accepts scalars or broadcastable arrays."""
    lat1r = np.radians(np.asarray(lat1, dtype=np.float64))
    lon1r = np.radians(np.asarray(lon1, dtype=np.float64))
    lat2r = np.radians(np.asarray(lat2, dtype=np.float64))
    lon2r = np.radians(np.asarray(lon2, dtype=np.float64))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def ground_truth_positive_ids(
    q_lat: float,
    q_lon: float,
    db_ids: np.ndarray,
    db_lat: np.ndarray,
    db_lon: np.ndarray,
    threshold_m: float = GT_THRESHOLD_M,
) -> set[int]:
    """Database ids whose GPS position is within ``threshold_m`` of the query (distance-only GT)."""
    dist_km = haversine_km(q_lat, q_lon, db_lat, db_lon)
    within = dist_km * 1000.0 <= threshold_m
    return {int(i) for i in db_ids[within]}


def query_hit_flags(ranked_ids: list[int], positives: set[int], ks: tuple[int, ...]) -> list[bool]:
    """Per-k correctness: True at k iff any of the top-k retrieved ids is a positive."""
    return [any(i in positives for i in ranked_ids[:k]) for k in ks]


def _read_allowed_query_keys(raw_dir: Path, city: str, subtask: str) -> set[str]:
    """Keys of the queries selected for ``subtask`` by ``query/subtask_index.csv`` (value True)."""
    path = raw_dir / "train_val" / city / "query" / "subtask_index.csv"
    if not path.exists():
        return set()
    allowed: set[str] = set()
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if str(row.get(subtask, "")).strip().lower() == "true":
                allowed.add(row["key"])
    return allowed


def _load_database(conn: sqlite3.Connection) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (db_ids, db_lat, db_lon) arrays for all ``image_role='database'`` rows."""
    rows = conn.execute(
        "SELECT id, lat, lon FROM reference_images WHERE image_role = 'database'"
    ).fetchall()
    ids = np.array([r["id"] for r in rows], dtype=np.int64)
    lat = np.array([r["lat"] for r in rows], dtype=np.float64)
    lon = np.array([r["lon"] for r in rows], dtype=np.float64)
    return ids, lat, lon


def _load_queries(
    conn: sqlite3.Connection, cities: tuple[str, ...], allowed_by_city: dict[str, set[str]]
) -> list[QueryRow]:
    """Return query rows (subtask-filtered) with an image on disk, for the given cities."""
    queries: list[QueryRow] = []
    for city in cities:
        allowed = allowed_by_city.get(city, set())
        rows = conn.execute(
            "SELECT reference_image_id, image_path, lat, lon FROM reference_images "
            "WHERE image_role = 'query' AND city = ?",
            (city,),
        ).fetchall()
        for r in rows:
            if r["reference_image_id"] not in allowed:
                continue
            if not Path(str(r["image_path"])).exists():
                continue
            queries.append(
                QueryRow(
                    key=r["reference_image_id"],
                    image_path=str(r["image_path"]),
                    lat=float(r["lat"]),
                    lon=float(r["lon"]),
                    city=city,
                )
            )
    return queries


def _embed_queries(
    embedding: EmbeddingService, queries: list[QueryRow], batch_size: int
) -> np.ndarray:
    """Embed every query image (in order) into a (Q, dim) matrix of unit descriptors."""
    vectors: list[np.ndarray] = []
    t0 = time.time()
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        images: list[Image.Image] = []
        for row in batch:
            with Image.open(row.image_path) as img:
                images.append(img.convert("RGB").copy())
        vectors.append(embedding.embed_batch(images))
        done = min(start + batch_size, len(queries))
        rate = done / max(time.time() - t0, 1e-9)
        print(f"    embedded {done}/{len(queries)} queries ({rate:.1f} img/s)", flush=True)
    return np.vstack(vectors)


def evaluate(
    index: faiss.Index,
    db_ids: np.ndarray,
    db_lat: np.ndarray,
    db_lon: np.ndarray,
    queries: list[QueryRow],
    query_vectors: np.ndarray,
    ks: tuple[int, ...],
    threshold_m: float,
) -> dict[str, object]:
    """Search each query against the index and compute Recall@k + km error metrics."""
    top_k = max(ks)
    _, retrieved = index.search(query_vectors, top_k)  # (Q, top_k) of db_ids
    id_to_pos = {int(i): p for p, i in enumerate(db_ids)}

    hit_counts = np.zeros(len(ks), dtype=np.int64)
    top1_errors_km: list[float] = []
    evaluated = 0
    for qi, query in enumerate(queries):
        positives = ground_truth_positive_ids(
            query.lat, query.lon, db_ids, db_lat, db_lon, threshold_m
        )
        if not positives:
            continue  # no reachable positive -> excluded from recall (official behaviour)
        evaluated += 1
        ranked = [int(i) for i in retrieved[qi] if i != -1]
        for j, hit in enumerate(query_hit_flags(ranked, positives, ks)):
            if hit:
                hit_counts[j] += 1
        pos = id_to_pos[ranked[0]]
        top1_errors_km.append(
            float(haversine_km(query.lat, query.lon, db_lat[pos], db_lon[pos]))
        )

    recall = {
        f"recall@{k}": (hit_counts[j] / evaluated if evaluated else 0.0)
        for j, k in enumerate(ks)
    }
    errors = np.array(top1_errors_km, dtype=np.float64)
    return {
        "n_queries_total": len(queries),
        "n_queries_evaluated": evaluated,
        "n_queries_no_positive": len(queries) - evaluated,
        "threshold_m": threshold_m,
        "ks": list(ks),
        **{k: round(float(v), 4) for k, v in recall.items()},
        "median_km_error": round(float(np.median(errors)), 4) if errors.size else None,
        "mean_km_error": round(float(np.mean(errors)), 4) if errors.size else None,
    }


def main() -> int:
    """Run the official MSLS-val evaluation and print / dump the metrics."""
    parser = argparse.ArgumentParser(description="MSLS-val evaluation (Recall@N + km error).")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threshold", type=float, default=GT_THRESHOLD_M, help="GT distance (m).")
    parser.add_argument("--subtask", default="all", help="subtask_index column selecting queries.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Cap #queries (smoke runs).")
    parser.add_argument("--city", action="append", help="Restrict to city (repeatable).")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON metrics path.")
    args = parser.parse_args()

    settings = get_settings()
    raw_dir = settings.msls_raw_dir or (settings.raw_dir / "msls_images")
    index_path = settings.index_dir / "index.faiss"
    db_path = settings.index_dir / "metadata.db"
    for path in (index_path, db_path):
        if not path.exists():
            print(f"error: missing artifact {path} (run build_index.py first)", file=sys.stderr)
            return 2
    if not raw_dir.exists():
        print(f"error: MSLS raw dir not found: {raw_dir}", file=sys.stderr)
        return 2

    cities = tuple(args.city) if args.city else settings.target_cities
    ks = DEFAULT_KS

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    db_ids, db_lat, db_lon = _load_database(conn)
    allowed_by_city = {c: _read_allowed_query_keys(raw_dir, c, args.subtask) for c in cities}
    queries = _load_queries(conn, cities, allowed_by_city)
    conn.close()

    if args.limit is not None:
        queries = queries[: args.limit]
    if not queries:
        print("error: no queries selected (check subtask / cities / data)", file=sys.stderr)
        return 2

    print(
        f"database={len(db_ids)} vectors | queries={len(queries)} "
        f"(subtask={args.subtask}, cities={','.join(cities)}) | device={args.device}",
        flush=True,
    )

    print(f"Loading VPR model on {args.device}...", flush=True)
    embedding = EmbeddingService.load(pretrained=True, device=args.device)
    index = read_index(index_path)

    query_vectors = _embed_queries(embedding, queries, args.batch_size)
    metrics = evaluate(index, db_ids, db_lat, db_lon, queries, query_vectors, ks, args.threshold)

    print("\n=== MSLS-val metrics ===", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    if args.output is not None:
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
