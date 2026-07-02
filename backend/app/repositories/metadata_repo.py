"""Reference-image metadata lookup by FAISS ``db_id``.

Each vector in the FAISS index carries a database id (via ``IndexIDMap``). To
turn a ``RetrievalHit`` into a geographic prediction the service must resolve
that id back to the reference image's coordinates and identifiers. This module
defines the lookup interface (``MetadataLookup``) plus the row shape
(``ReferenceMeta``).

The real SQLite-backed implementation is built by ``scripts/build_index.py`` in
Sprint S1 (writes the ``reference_images`` table alongside the index). For the
data-independent S2 skeleton and unit tests an in-memory implementation
(``InMemoryMetadataRepo``) is provided.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ReferenceMeta:
    """Geographic + identity metadata for one indexed reference image."""

    db_id: int
    lat: float
    lon: float
    reference_image_id: str
    sequence_id: str | None = None


class MetadataLookup(Protocol):
    """Resolve FAISS ``db_id`` values back to reference-image metadata."""

    def get(self, db_id: int) -> ReferenceMeta | None:
        """Return metadata for ``db_id`` or ``None`` if unknown."""
        ...


class InMemoryMetadataRepo:
    """Dict-backed ``MetadataLookup`` for the S2 skeleton and tests."""

    def __init__(self, rows: list[ReferenceMeta]) -> None:
        """Index ``rows`` by their ``db_id`` for O(1) lookup."""
        self._by_id: dict[int, ReferenceMeta] = {row.db_id: row for row in rows}

    def get(self, db_id: int) -> ReferenceMeta | None:
        """Return metadata for ``db_id`` or ``None`` if unknown."""
        return self._by_id.get(db_id)


class SqliteMetadataRepo:
    """SQLite-backed ``MetadataLookup`` over the ``reference_images`` table.

    Only ``image_role='database'`` rows are addressable here: their ``id`` is the
    FAISS ``IndexIDMap`` id, so a retrieval hit resolves to exactly one row. Query
    rows live in the same table (for offline evaluation) but are never returned by
    ``get`` because their ids are not present in the index. The connection is
    opened read-only and ``check_same_thread=False`` so it is safe to share across
    the ``run_in_threadpool`` workers that serve predictions.
    """

    def __init__(self, db_path: Path) -> None:
        """Open a read-only connection to the metadata database at ``db_path``."""
        uri = f"file:{db_path.as_posix()}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def get(self, db_id: int) -> ReferenceMeta | None:
        """Return database-row metadata for ``db_id`` or ``None`` if unknown."""
        row = self._conn.execute(
            "SELECT id, lat, lon, reference_image_id, sequence_id "
            "FROM reference_images WHERE id = ? AND image_role = 'database'",
            (db_id,),
        ).fetchone()
        if row is None:
            return None
        return ReferenceMeta(
            db_id=row["id"],
            lat=row["lat"],
            lon=row["lon"],
            reference_image_id=row["reference_image_id"],
            sequence_id=row["sequence_id"],
        )

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
