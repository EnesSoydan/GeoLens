"""Offline pipeline: embed database images -> FAISS IndexIDMap + SQLite metadata.

Reads the extracted MSLS-val layout::

    <msls_raw>/train_val/<city>/<role>/images/<key>.jpg
    <msls_raw>/train_val/<city>/<role>/raw.csv       # key, lon, lat, ca, ...
    <msls_raw>/train_val/<city>/<role>/seq_info.csv  # key, sequence_key, ...

with ``role in {database, query}`` and ``city`` in the target list (cph, sf).

Data-leakage prevention (docs/architecture/03-ai-pipeline.md, 06-veri-modeli.md):
**only** ``image_role='database'`` vectors are written to the FAISS index; query
rows are stored in the SQLite ``reference_images`` table for offline evaluation
(``evaluate.py``) but never enter the index, so a query can never retrieve itself.

Outputs (under ``settings.index_dir``):
* ``index.faiss``   - IndexIDMap(IndexFlatIP) over unit descriptors, id = row id
* ``metadata.db``   - SQLite ``reference_images`` (database + query rows)
* ``manifest.json`` - ModelManifest sidecar (index_size = #database vectors)

Usage::

    PYTHONPATH=backend python scripts/build_index.py [--limit N] [--city cph]
        [--batch-size 32] [--device cpu|cuda]

``--limit`` caps the number of database images per city for a fast smoke build.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
from PIL import Image, UnidentifiedImageError

# scripts/ is run with PYTHONPATH=backend so ``app`` is importable.
from app.core.config import get_settings
from app.ml.aggregator import DESCRIPTOR_DIM
from app.ml.faiss_io import write_index
from app.ml.model_registry import default_manifest
from app.services.embedding import EmbeddingService

ROLES = ("database", "query")


def _read_city_role(raw_dir: Path, city: str, role: str) -> list[dict[str, object]]:
    """Join raw.csv (GPS) + seq_info.csv (sequence) for one city/role folder.

    Returns one dict per image key that has both metadata and an image file on
    disk (missing images are skipped, e.g. when ``--limit`` extracted a subset).
    """
    base = raw_dir / "train_val" / city / role
    raw_csv = base / "raw.csv"
    seq_csv = base / "seq_info.csv"
    images_dir = base / "images"
    if not raw_csv.exists():
        return []

    sequences: dict[str, str] = {}
    if seq_csv.exists():
        with seq_csv.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                sequences[row["key"]] = row["sequence_key"]

    rows: list[dict[str, object]] = []
    with raw_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = row["key"]
            image_path = images_dir / f"{key}.jpg"
            if not image_path.exists():
                continue
            rows.append(
                {
                    "reference_image_id": key,
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "ca": float(row["ca"]) if row.get("ca") else None,
                    "sequence_id": sequences.get(key),
                    "city": city,
                    "image_role": role,
                    "image_path": str(image_path),
                }
            )
    return rows


def _init_db(db_path: Path) -> sqlite3.Connection:
    """(Re)create the ``reference_images`` table and return an open connection."""
    db_path.unlink(missing_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE reference_images (
            id                 INTEGER PRIMARY KEY,
            reference_image_id TEXT    NOT NULL,
            image_path         TEXT    NOT NULL,
            lat                REAL    NOT NULL,
            lon                REAL    NOT NULL,
            ca                 REAL,
            sequence_id        TEXT,
            city               TEXT    NOT NULL,
            source             TEXT    NOT NULL DEFAULT 'MSLS',
            split              TEXT    NOT NULL DEFAULT 'val',
            image_role         TEXT    NOT NULL,
            created_at         TEXT    NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX idx_role ON reference_images(image_role)")
    return conn


def _insert_rows(
    conn: sqlite3.Connection, rows: list[dict[str, object]], start_id: int
) -> list[int]:
    """Insert rows with sequential ids from ``start_id``; return the assigned ids."""
    created_at = datetime.now(UTC).isoformat()
    ids = list(range(start_id, start_id + len(rows)))
    conn.executemany(
        """
        INSERT INTO reference_images
            (id, reference_image_id, image_path, lat, lon, ca, sequence_id,
             city, source, split, image_role, created_at)
        VALUES (:id, :reference_image_id, :image_path, :lat, :lon, :ca,
                :sequence_id, :city, 'MSLS', 'val', :image_role, :created_at)
        """,
        [{**r, "id": i, "created_at": created_at} for i, r in zip(ids, rows, strict=True)],
    )
    return ids


def _embed_and_add(
    embedding: EmbeddingService,
    index: faiss.IndexIDMap,
    rows: list[dict[str, object]],
    ids: list[int],
    batch_size: int,
) -> int:
    """Embed database images in batches and add unit vectors under their ids."""
    added = 0
    t0 = time.time()
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch_ids = ids[start : start + batch_size]
        images: list[Image.Image] = []
        keep_ids: list[int] = []
        for row, db_id in zip(batch_rows, batch_ids, strict=True):
            try:
                with Image.open(row["image_path"]) as img:  # type: ignore[arg-type]
                    images.append(img.convert("RGB").copy())
                keep_ids.append(db_id)
            except (UnidentifiedImageError, OSError):
                continue
        if not images:
            continue
        vectors = embedding.embed_batch(images)
        index.add_with_ids(vectors, np.asarray(keep_ids, dtype=np.int64))
        added += len(images)
        rate = added / max(time.time() - t0, 1e-9)
        print(f"    embedded {added}/{len(rows)} ({rate:.1f} img/s)", flush=True)
    return added


def main() -> int:
    """Build the FAISS index + SQLite metadata + manifest from MSLS-val."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Max database images per city.")
    parser.add_argument("--city", action="append", help="Restrict to city (repeatable).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    settings = get_settings()
    raw_dir = settings.msls_raw_dir or (settings.raw_dir / "msls_images")
    if not raw_dir.exists():
        print(f"error: MSLS raw dir not found: {raw_dir}", file=sys.stderr)
        return 2

    cities = tuple(args.city) if args.city else settings.target_cities
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    index_path = settings.index_dir / "index.faiss"
    db_path = settings.index_dir / "metadata.db"
    manifest_path = settings.index_dir / "manifest.json"

    print(f"Loading VPR model on {args.device}...", flush=True)
    embedding = EmbeddingService.load(pretrained=True, device=args.device)

    index = faiss.IndexIDMap(faiss.IndexFlatIP(DESCRIPTOR_DIM))
    conn = _init_db(db_path)
    next_id = 0
    total_db = 0

    for city in cities:
        for role in ROLES:
            rows = _read_city_role(raw_dir, city, role)
            if role == "database" and args.limit is not None:
                rows = rows[: args.limit]
            if not rows:
                continue
            ids = _insert_rows(conn, rows, next_id)
            next_id += len(rows)
            print(f"[{city}/{role}] rows={len(rows)}", flush=True)
            if role == "database":
                total_db += _embed_and_add(embedding, index, rows, ids, args.batch_size)

    conn.commit()
    conn.close()

    write_index(index, index_path)
    manifest = default_manifest(index_size=index.ntotal)
    manifest.to_json(manifest_path)

    print(
        f"\nDONE: index_size={index.ntotal} db_rows_embedded={total_db} "
        f"total_sqlite_rows={next_id}\n  index={index_path}\n  db={db_path}\n"
        f"  manifest={manifest_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
