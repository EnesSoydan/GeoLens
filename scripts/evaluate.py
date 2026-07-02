"""Evaluation: official MSLS-val protocol (Recall@N + km error).

Implemented in Sprint S5. Uses the official mapillary_sls protocol (740 queries
over the MSLS-val cities Copenhagen + San Francisco; GT threshold 25 m AND
<=40 deg viewpoint), plus a NetVLAD baseline on the same subset. See
docs/architecture/08-degerlendirme.md.
"""

from __future__ import annotations

import sys


def main() -> int:
    """Entry point (implemented in Sprint S5)."""
    raise SystemExit("evaluate is implemented in Sprint S5.")


if __name__ == "__main__":
    sys.exit(main())
