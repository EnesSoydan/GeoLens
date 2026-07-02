"""Filter the MSLS-val subset used by GeoLens.

The Mapillary Street-Level Sequences (MSLS) dataset is license-gated
(CC-BY-SA, Mapillary registration required). This script does NOT bypass that:
it expects you to have already obtained the MSLS ``train_val`` archive from
https://www.mapillary.com/dataset/places and only *filters* the official
validation cities (default: Copenhagen ``cph`` + San Francisco ``sf``) into
``data/raw/msls_val/<city>``. Each city keeps its own ``database``/``query``
layout so that the 740-query MSLS-val protocol can be reproduced downstream.

Parsing the filtered subset into SQLite + FAISS happens later in Sprint S1
(``scripts/build_index.py``).

Usage:
    python scripts/download_msls.py --source /path/to/msls_root
    python scripts/download_msls.py --source /path/to/msls_train_val.zip
    python scripts/download_msls.py --source /path/to/msls_root --cities cph
    python scripts/download_msls.py --source /path/to/msls_root --dry-run
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("download_msls")

DATASET_PAGE = "https://www.mapillary.com/dataset/places"
# Official mapillary_sls validation split (public ground truth).
DEFAULT_CITIES = ("cph", "sf")
DEFAULT_DEST = Path("data/raw/msls_val")
# A valid MSLS city dir contains at least one of these role sub-directories.
CITY_CHILD_MARKERS = ("database", "query")


@dataclass
class CopyStats:
    """Aggregate counts for a copy/extract run."""

    files: int = 0
    skipped: int = 0
    roles: dict[str, int] = field(default_factory=dict)


def _has_city_markers(path: Path) -> bool:
    """True if ``path`` looks like an MSLS city dir (has database/query)."""
    return any((path / marker).is_dir() for marker in CITY_CHILD_MARKERS)


def find_city_dir(source_root: Path, city: str) -> Path | None:
    """Locate the MSLS city directory under ``source_root``.

    Args:
        source_root: Extracted MSLS root (any depth above the city dir).
        city: Target city name (case-insensitive).

    Returns:
        The city directory containing ``database``/``query``, or ``None``.
    """
    city = city.lower()
    if source_root.name.lower() == city and _has_city_markers(source_root):
        return source_root
    for path in source_root.rglob("*"):
        if path.is_dir() and path.name.lower() == city and _has_city_markers(path):
            return path
    return None


def copy_city_subset(city_dir: Path, dest: Path, overwrite: bool, dry_run: bool) -> CopyStats:
    """Copy every file under ``city_dir`` into ``dest`` preserving structure."""
    stats = CopyStats()
    for src in sorted(city_dir.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(city_dir)
        role = rel.parts[0] if rel.parts else "root"
        stats.roles[role] = stats.roles.get(role, 0) + 1
        target = dest / rel
        if target.exists() and not overwrite:
            stats.skipped += 1
            continue
        stats.files += 1
        if dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    return stats


def extract_city_from_zip(zip_path: Path, city: str, dest: Path, dry_run: bool) -> CopyStats:
    """Extract only ``city`` members from an MSLS ``.zip`` archive."""
    city = city.lower()
    needle = f"/{city}/"
    stats = CopyStats()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            haystack = f"/{info.filename.lower()}"
            if needle not in haystack:
                continue
            idx = haystack.index(needle) + len(needle)
            rel = Path(info.filename[idx - 1 :].lstrip("/"))
            if not rel.parts:
                continue
            stats.roles[rel.parts[0]] = stats.roles.get(rel.parts[0], 0) + 1
            stats.files += 1
            if dry_run:
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as sf, open(target, "wb") as tf:
                shutil.copyfileobj(sf, tf)
    return stats


def _print_access_instructions() -> None:
    """Explain how to obtain the license-gated dataset."""
    logger.error(
        "No --source provided. MSLS is license-gated and cannot be "
        "auto-downloaded.\n"
        "  1. Register + accept the license at %s\n"
        "  2. Download the train_val archive.\n"
        "  3. Re-run: python scripts/download_msls.py "
        "--source <extracted_dir_or_zip>",
        DATASET_PAGE,
    )


def _run_city(source: Path, city: str, city_dest: Path, overwrite: bool, dry_run: bool) -> int:
    """Filter a single city into ``city_dest`` and return an exit code."""
    if source.is_file() and source.suffix.lower() == ".zip":
        stats = extract_city_from_zip(source, city, city_dest, dry_run)
    else:
        city_dir = find_city_dir(source, city)
        if city_dir is None:
            logger.error("City '%s' not found under %s", city, source)
            return 1
        logger.info("Found city dir: %s", city_dir)
        stats = copy_city_subset(city_dir, city_dest, overwrite, dry_run)

    if stats.files == 0 and stats.skipped == 0:
        logger.error("No files matched city '%s'.", city)
        return 1

    verb = "Would copy" if dry_run else "Copied"
    logger.info("%s %d files (skipped %d) -> %s", verb, stats.files, stats.skipped, city_dest)
    for role, count in sorted(stats.roles.items()):
        logger.info("  role=%s: %d files", role, count)
    return 0


def run(
    source: Path | None,
    cities: list[str],
    dest: Path,
    overwrite: bool,
    dry_run: bool,
) -> int:
    """Filter every city in ``cities`` into ``dest/<city>`` and return an exit code."""
    if source is None:
        _print_access_instructions()
        return 2
    if not source.exists():
        logger.error("Source not found: %s", source)
        return 2

    dest.mkdir(parents=True, exist_ok=True)
    for city in cities:
        city_dest = dest / city.lower()
        city_dest.mkdir(parents=True, exist_ok=True)
        code = _run_city(source, city, city_dest, overwrite, dry_run)
        if code != 0:
            return code
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter the MSLS-val (cph+sf) subset into data/raw."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Extracted MSLS root directory or a .zip archive.",
    )
    parser.add_argument(
        "--cities",
        default=",".join(DEFAULT_CITIES),
        help="Comma-separated target cities (default: cph,sf).",
    )
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Destination directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--dry-run", action="store_true", help="Report counts without copying.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    return run(args.source, cities, args.dest, args.overwrite, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
