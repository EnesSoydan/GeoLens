"""S-V2-1 shard-alignment probe for OSV5M.

The CSV is geographically sorted (osv5m_geo_probe.py). The open question: does a
*shard* (the zip ``images/train/00.zip``) correspond to a contiguous CSV block
(-> geographically clustered) or is its membership independent of CSV order
(-> globally mixed)?

Definitive, cheap test: read shard-00's id list from the zip central directory
via HTTP Range (no 2.5 GB download), read the first ~50 001 CSV rows (~40 MB),
and measure how many shard-00 ids fall in that first CSV block.

  overlap ~= 100 %  -> shard 00 == first CSV block -> shards are CLUSTERED
  overlap ~=   1 %  -> uniform (50001/4.89M base rate) -> shards are MIXED

Prints a compact report only.
"""

from __future__ import annotations

import argparse
import io
import struct

import pandas as pd
import requests

CSV_URL = "https://huggingface.co/datasets/osv5m/osv5m/resolve/main/train.csv"
SHARD_URL_TMPL = (
    "https://huggingface.co/datasets/osv5m/osv5m/resolve/main/images/train/{shard}.zip"
)
FIRST_BLOCK_ROWS = 50_001  # one shard's worth of rows
FIRST_BLOCK_BYTES = 45_000_000  # ~45 MB, enough to cover 50k rows
TRAIN_TOTAL = 4_894_685


def _range(url: str, start: int, end: int) -> bytes:
    """Fetch bytes [start, end] via an HTTP Range request (HF signs per range)."""
    r = requests.get(
        url, headers={"Range": f"bytes={start}-{end}"}, allow_redirects=True, timeout=120
    )
    r.raise_for_status()
    return r.content


def _content_length(url: str) -> int:
    """Return the file size in bytes."""
    r = requests.head(url, allow_redirects=True, timeout=30)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def _shard_ids(shard_url: str) -> set[str]:
    """Read the zip central directory over HTTP Range (no 2.5 GB download).

    The shard is a <4 GB, <65535-entry zip, so the classic End-Of-Central-Directory
    record (no zip64) applies: read the tail to find the EOCD, then range-read the
    central directory and parse each file header's name.
    """
    size = _content_length(shard_url)
    tail = _range(shard_url, max(0, size - 65_557), size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("EOCD imzasi bulunamadi")
    cd_size, cd_offset = struct.unpack_from("<II", tail, eocd + 12)
    cd = _range(shard_url, cd_offset, cd_offset + cd_size - 1)

    ids: set[str] = set()
    pos = 0
    while pos + 46 <= len(cd) and cd[pos : pos + 4] == b"PK\x01\x02":
        name_len, extra_len, comment_len = struct.unpack_from("<HHH", cd, pos + 28)
        name = cd[pos + 46 : pos + 46 + name_len].decode("utf-8", "ignore")
        if name.lower().endswith((".jpg", ".jpeg")):
            ids.add(name.split("/")[-1].rsplit(".", 1)[0])
        pos += 46 + name_len + extra_len + comment_len
    return ids


def _first_block() -> pd.DataFrame:
    """Return id/lat/lon/country for the first ~50 001 CSV rows via HTTP Range."""
    r = requests.get(
        CSV_URL, headers={"Range": f"bytes=0-{FIRST_BLOCK_BYTES - 1}"}, timeout=120
    )
    r.raise_for_status()
    text = r.content.decode("utf-8", errors="ignore")
    # Drop the last (possibly truncated) line before parsing.
    text = text[: text.rfind("\n")]
    df = pd.read_csv(
        io.StringIO(text),
        usecols=lambda c: c in ("id", "latitude", "longitude", "country"),
        on_bad_lines="skip",
    )
    return df.head(FIRST_BLOCK_ROWS)


def main() -> int:
    """Measure a shard's id overlap with the first CSV block and print a verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard",
        default="00",
        help="shard number, zero-padded 2 digits (e.g. 00, 49, 97)",
    )
    args = parser.parse_args()
    shard = args.shard
    shard_url = SHARD_URL_TMPL.format(shard=shard)

    print(f"[zip] shard-{shard} id listesi (merkezi dizin, indirmesiz) okunuyor...")
    sids = _shard_ids(shard_url)
    print(f"[zip] shard-{shard} goruntu sayisi: {len(sids)}")

    print("[csv] ilk ~50k satir (Range ~45MB) okunuyor...")
    df = _first_block()
    block_ids = set(df["id"].astype(str))
    print(f"[csv] ilk blok satir: {len(block_ids)}")

    inter = sids & block_ids
    overlap = len(inter) / len(sids) if sids else 0.0
    base_rate = FIRST_BLOCK_ROWS / TRAIN_TOTAL
    print(f"[test] shard-{shard} kesisim ilk-blok: {len(inter)} ({overlap:.1%})")
    print(f"[test] uniform taban orani: {base_rate:.1%}")
    if overlap > 0.5:
        verdict = "KUMELI (shard ~= CSV blogu; ilk N shard cografi yanli)"
    elif overlap < 3 * base_rate:
        verdict = "KARISIK (shard uyeligi CSV sirasindan bagimsiz; ilk N shard ~kuresel)"
    else:
        verdict = "BELIRSIZ (kismi ortusme; ek analiz gerek)"
    print(f"[VERDICT] {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
