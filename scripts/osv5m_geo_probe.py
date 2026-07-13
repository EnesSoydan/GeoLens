"""S-V2-1 geographic-distribution probe for OSV5M shards.

Question: are OSV5M shards geographically *balanced* (each shard ~a uniform
global sample) or *clustered* (shards grouped by region)? The answer decides the
sampling strategy: if balanced, downloading the first N shards yields a global
subset cheaply; if clustered, we must spread the sample across all 98 shards.

Rather than downloading the 2.9 GB ``train.csv``, this reads three small windows
(~3 MB each) at ~0 %, ~50 % and ~99 % of the file via HTTP Range, parses the rows,
and measures the spread of latitude/longitude and countries in each window. It
also checks whether the known shard-00 image ids fall in the first window, i.e.
whether CSV row order matches shard order.

Prints a compact report only. Downloads ~9 MB total.
"""

from __future__ import annotations

import io

import pandas as pd
import requests

URL = "https://huggingface.co/datasets/osv5m/osv5m/resolve/main/train.csv"
WINDOW = 3_000_000  # ~3 MB per range read
FRACTIONS = [0.0, 0.5, 0.99]
# Known shard-00 image ids observed in S-V2-0 (images/train/00.zip).
KNOWN_SHARD0_IDS = {"181393113831089", "346368943791346", "148306984644239"}


def _total_size() -> int:
    """Return train.csv size in bytes from a HEAD request."""
    r = requests.head(URL, allow_redirects=True, timeout=30)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def _range(start: int, length: int) -> bytes:
    """Fetch ``length`` bytes starting at ``start`` via an HTTP Range request."""
    end = start + length - 1
    r = requests.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
    r.raise_for_status()
    return r.content


def _coarse_cells(df: pd.DataFrame) -> int:
    """Count distinct ~10-degree lat/lon grid cells (proxy for global spread)."""
    cells = set(
        zip(
            (df["latitude"] // 10).astype(int),
            (df["longitude"] // 10).astype(int),
            strict=False,
        )
    )
    return len(cells)


def main() -> int:
    """Probe three CSV windows and report geographic spread per window."""
    size = _total_size()
    print(f"[csv] train.csv boyutu: {size / 1e9:.2f} GB")

    header_text = _range(0, WINDOW).decode("utf-8", errors="ignore")
    header = header_text.splitlines()[0].split(",")
    print(f"[csv] kolonlar: {header}")
    has_country = "country" in header

    for frac in FRACTIONS:
        start = int(size * frac)
        raw = _range(start, WINDOW).decode("utf-8", errors="ignore")
        lines = raw.splitlines()
        if frac > 0.0:
            lines = lines[1:]  # drop partial first line at a mid-file offset
        text = "\n".join(lines)
        df = pd.read_csv(
            io.StringIO(text),
            names=header,
            header=0 if frac == 0.0 else None,
            usecols=lambda c: c in ("id", "latitude", "longitude", "country"),
            on_bad_lines="skip",
        )
        df = df.dropna(subset=["latitude", "longitude"])
        cells = _coarse_cells(df)
        lat_span = df["latitude"].max() - df["latitude"].min()
        lon_span = df["longitude"].max() - df["longitude"].min()
        ncountry = df["country"].nunique() if has_country else -1
        print(
            f"[pencere {frac:>4.0%}] satir={len(df):5d} "
            f"grid-hucre(~10deg)={cells:3d} "
            f"ulke={ncountry:3d} "
            f"lat-span={lat_span:6.1f} lon-span={lon_span:6.1f}"
        )
        if frac == 0.0:
            ids = set(df["id"].astype(str))
            hit = KNOWN_SHARD0_IDS & ids
            print(f"[sira] bilinen shard-00 id'leri ilk pencerede: {len(hit)}/3 {sorted(hit)}")
            print(f"[pencere 0%] ornek ulkeler: {sorted(df['country'].dropna().unique())[:12]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
