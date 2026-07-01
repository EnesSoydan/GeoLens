"""Offline pipeline: embed database images -> FAISS IndexIDMap + SQLite.

Implemented in Sprint S1. Only ``image_role='database'`` images are written to
the FAISS index; ``query`` images are stored in SQLite for evaluation only
(data-leakage prevention, see docs/architecture/03-ai-pipeline.md).
"""

from __future__ import annotations

import sys


def main() -> int:
    """Entry point (implemented in Sprint S1)."""
    raise SystemExit("build_index is implemented in Sprint S1.")


if __name__ == "__main__":
    sys.exit(main())
