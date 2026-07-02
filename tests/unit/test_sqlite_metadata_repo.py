"""Unit tests for the SQLite-backed metadata repository (Sprint S1)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.repositories.metadata_repo import SqliteMetadataRepo


def _build_db(path: Path) -> None:
    """Create a minimal reference_images table with one database + one query row."""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE reference_images (
            id INTEGER PRIMARY KEY,
            reference_image_id TEXT NOT NULL,
            image_path TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            ca REAL,
            sequence_id TEXT,
            city TEXT NOT NULL,
            source TEXT NOT NULL,
            split TEXT NOT NULL,
            image_role TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.executemany(
        "INSERT INTO reference_images VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (0, "dbkey", "cph/database/images/dbkey.jpg", 55.69, 12.56, 235.8,
             "seqA", "cph", "MSLS", "val", "database", "2026-01-01T00:00:00Z"),
            (1, "qkey", "cph/query/images/qkey.jpg", 55.70, 12.57, 221.7,
             "seqB", "cph", "MSLS", "val", "query", "2026-01-01T00:00:00Z"),
        ],
    )
    conn.commit()
    conn.close()


@pytest.fixture
def repo(tmp_path: Path) -> SqliteMetadataRepo:
    db_path = tmp_path / "metadata.db"
    _build_db(db_path)
    return SqliteMetadataRepo(db_path)


def test_get_returns_database_row(repo: SqliteMetadataRepo) -> None:
    meta = repo.get(0)
    assert meta is not None
    assert meta.db_id == 0
    assert meta.reference_image_id == "dbkey"
    assert meta.sequence_id == "seqA"
    assert meta.lat == pytest.approx(55.69)
    assert meta.lon == pytest.approx(12.56)


def test_query_row_is_not_addressable(repo: SqliteMetadataRepo) -> None:
    # Data-leakage guard: query rows exist in SQLite but are never resolvable via
    # get() (their ids are not in the FAISS index; role filter excludes them).
    assert repo.get(1) is None


def test_unknown_id_returns_none(repo: SqliteMetadataRepo) -> None:
    assert repo.get(999) is None
