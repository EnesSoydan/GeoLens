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

from dataclasses import dataclass
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
